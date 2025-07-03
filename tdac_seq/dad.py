import numpy as np
from tqdm import tqdm
from hmmlearn import hmm
import logging
from Bio.Seq import reverse_complement
import scipy.sparse
import itertools

def call_dads_merge(edits: np.ndarray, distance_thresh: int = 80) -> np.ndarray:
    '''
    Given an array of edited bools, return an array of bools indicating dad regions. Input should have shape (reads, sequence length). Adapted from Stergachis et al. (2020), a DAD is defined as a region of the read where two or more edits are within `distance_thresh` bp of each other. They used 65 bp as the threshold
    '''
    logging.warning("Deprecated. Prefer call_dads_hmm")
    dad = np.zeros_like(edits, dtype=bool)
    for read_i, read in tqdm(enumerate(edits), total=len(edits), leave=False, desc="Calling DADs"):
        prev_edit_pos = None
        for pos, edited in enumerate(read):
            if edited:
                if prev_edit_pos is not None and pos - prev_edit_pos < distance_thresh:
                    dad[read_i, prev_edit_pos:pos] = True
                prev_edit_pos = pos
    return dad

def call_dads_hmm(edits: np.ndarray, ref_seq: str, strands: np.ndarray, start_gap_threshold: int = 500, end_gap_threshold: int = 500) -> np.ndarray:
    """
    `edits` is a boolean array of shape (reads, sequence length) where True indicates an edit. `ref_seq` is the reference sequence. `strands` is an array of shape (reads,) where 0 indicates the read is from the C-to-T strand and 1 indicates the read is from the G-to-A strand. `start_gap_threshold` and `end_gap_threshold` are the number of bases to ignore at the start and end of the reference sequence, respectively. Return a boolean array of shape (reads, sequence length) where True indicates a DAD region.
    """
    dad = np.zeros_like(edits, dtype=bool)
    for strand_i, strand_editfrom in [(0, "C"), (1, "G")]:
        editable = np.array([i for i, b in enumerate(ref_seq) if b == strand_editfrom and i > start_gap_threshold and i < len(ref_seq) - end_gap_threshold])
        edits_only = np.array(edits[strands == strand_i, :][:, editable].astype(int))
        model = hmm.CategoricalHMM(n_components=2, init_params='', verbose=True, n_iter=50)
        model.startprob_ = np.array([0.99, 0.01])
        model.transmat_ = np.array([[0.99683402, 0.00316598], [0.09315898, 0.90684102]])
        model.emissionprob_ = np.array([[0.98408976, 0.01591024], [0.52864081, 0.47135919]])
        # model.fit(edits[:, editable].reshape(-1, 1), lengths=np.full(len(edits), len(editable)))
        # model.startprob_
        # model.transmat_
        # model.emissionprob_
        prob = model.predict_proba(edits_only.reshape(-1, 1), lengths=np.full(edits_only.shape[0], edits_only.shape[1]))
        prob = prob[..., 1].reshape(*edits_only.shape)
        dad_only = np.zeros((prob.shape[0], edits.shape[1]), dtype=bool)
        prev_in_dad_j = None
        for i, read in enumerate(prob > 0.5):
            for j, in_dad in zip(editable, read):
                if in_dad and prev_in_dad_j:
                    dad_only[i, prev_in_dad_j:j] = True
                if in_dad:
                    prev_in_dad_j = j
                else:
                    prev_in_dad_j = None
        dad[strands == strand_i] = dad_only
    return dad

def _call_dads_hmm_bias(edits: np.ndarray, ref_seq: str, strands: np.ndarray, start_gap_threshold: int = 500, end_gap_threshold: int = 500) -> np.ndarray:
    dad = scipy.sparse.lil_array(edits.shape, dtype=float)
    for strand_i in range(2):
        edits_only = np.array(edits[strands == strand_i, :].astype(int))
        if strand_i == 1:
            edits_only = np.flip(edits_only, axis=1)
        model = FancyCategoricalHMM(ref_seq=ref_seq if strand_i == 0 else reverse_complement(ref_seq), n_components=2, init_params='e', verbose=True, n_iter=50, n_features=2)
        model.startprob_ = np.array([0.99, 0.01])
        model.transmat_ = np.array([[0.99458711, 0.00541289], [0.03575459, 0.96424541]])
        # model.emissionprob_ = np.array([[0.98408976, 0.01591024], [0.52864081, 0.47135919]])
        model.emissionprob_open_ = np.array([
            [0.48087695, 0.51912305],
            [0.9455473,  0.0544527 ],
            [0.79289521, 0.20710479],
            [0.91500725, 0.08499275],
        ])
        model.emissionprob_closed_ = np.array([0.99389325, 0.00610675])
        # Z_hat = model.fit(edits_only)
        logging.info(f"Strand {strand_i} model startprob: {model.startprob_}")
        logging.info(f"Strand {strand_i} model transmat: {model.transmat_}")
        logging.info(f"Strand {strand_i} model emissionprob_open: {model.emissionprob_open_}")
        logging.info(f"Strand {strand_i} model emissionprob_closed: {model.emissionprob_closed_}")
        original_indices = np.arange(len(edits_only))
        # break into minibatches
        for batch_indices in map(np.array, itertools.batched(original_indices, 1000)):
            Z_hat = model.predict(edits_only[batch_indices]).reshape(-1, edits_only.shape[1])
            if strand_i == 1:
                Z_hat = np.flip(Z_hat, axis=1)
            dad[np.nonzero(strands == strand_i)[0][batch_indices]] = Z_hat
    return dad

def call_dads_hmm_bias(edits: np.ndarray, ref_seq: str, strands: np.ndarray, start_gap_threshold: int = 500, end_gap_threshold: int = 500) -> np.ndarray:
    return (_call_dads_hmm_bias(edits, ref_seq, strands, start_gap_threshold, end_gap_threshold).tocsc() > 0).toarray()

def dad_probability_hmm(edits: np.ndarray, editable: np.ndarray):
    '''
    Edits be for a single sequence. Return the probability of being in a dad region for each position
    '''
    edits_only = np.array(edits[editable][np.newaxis].astype(int))
    model = hmm.CategoricalHMM(n_components=2, init_params='')
    model.startprob_ = np.array([0.99, 0.01])
    model.transmat_ = np.array([[0.99683402, 0.00316598], [0.09315898, 0.90684102]])
    model.emissionprob_ = np.array([[0.98408976, 0.01591024], [0.52864081, 0.47135919]])
    # model.fit(edits[:, editable].reshape(-1, 1), lengths=np.full(len(edits), len(editable)))
    # model.startprob_
    # model.transmat_
    # model.emissionprob_
    prob = model.predict_proba(edits_only.reshape(-1, 1), lengths=np.full(edits_only.shape[0], edits_only.shape[1]))
    prob = prob[:, 1]
    dad_prob = np.zeros(len(edits), dtype=float)
    for i in range(len(editable) - 1):
        dad_prob[editable[i]:editable[i + 1]] = prob[i]
    dad_prob[editable[-1]:] = prob[-1]
    return dad_prob

from hmmlearn.base import BaseHMM, _AbstractHMM
from hmmlearn.utils import normalize
import numpy as np
from sklearn.utils.validation import check_random_state, check_is_fitted
import warnings
import inspect
class FancyCategoricalHMM(BaseHMM, _AbstractHMM):
    """
    Hidden Markov Model with categorical (discrete) emissions.

    Attributes
    ----------
    n_features : int
        Number of possible symbols emitted by the model (in the samples).

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    emissionprob_open_ : array, shape (n_components, n_features)
        Probability of emitting a given symbol when in the open state 
        in TC, GC, CC, or AC sequence context.

    emissionprob_closed_ : array, shape (n_components, n_features)
        Probability of emitting a given symbol when in the closed state.

    Examples
    --------
    >>> from hmmlearn.hmm import CategoricalHMM
    >>> CategoricalHMM(n_components=2)  #doctest: +ELLIPSIS
    CategoricalHMM(algorithm='viterbi',...
    """

    def __init__(self, ref_seq: str, n_components=1, startprob_prior=1.0,
                 transmat_prior=1.0, *, emissionprob_prior=1.0,
                 n_features=None, algorithm="viterbi",
                 random_state=None, n_iter=10, tol=1e-2,
                 verbose=False, params="ste", init_params="ste",
                 implementation="log"):
        """
        Parameters
        ----------
        n_components : int
            Number of states.

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        emissionprob_prior : array, shape (n_components, n_features), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`emissionprob_`.

        n_features: int, optional
            The number of categorical symbols in the HMM.  Will be inferred
            from the data if not set.

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

            - "viterbi": finds the most likely sequence of states, given all
              emissions.
            - "map" (also known as smoothing or forward-backward): finds the
              sequence of the individual most-likely states, given all
              emissions.

        random_state: RandomState or an int seed, optional
            A random number generator instance.

        n_iter : int, optional
            Maximum number of iterations to perform.

        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.

        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.

        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any
            combination of 's' for startprob, 't' for transmat, and 'e' for
            emissionprob.  Defaults to all parameters.

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.
        """
        BaseHMM.__init__(self, n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior,
                         algorithm=algorithm,
                         random_state=random_state,
                         n_iter=n_iter, tol=tol, verbose=verbose,
                         params=params, init_params=init_params,
                         implementation=implementation)
        self.emissionprob_prior = emissionprob_prior
        self.n_features = n_features
        self.ref_seq = ref_seq
        self._sequence_contexts = {
            dinucleotide: np.array([i for i in range(len(self.ref_seq)) if self.ref_seq[i-1:i+1] == dinucleotide])
            for dinucleotide in ["TC", "GC", "CC", "AC"]
        }

    def _init(self, X, lengths=None):
        if lengths is not None:
            warnings.warn("The 'lengths' parameter is set but not used")
        lengths = np.full(X.shape[0], X.shape[1])
        super()._init(X, lengths)

        self.random_state = check_random_state(self.random_state)

        if self._needs_init('e', 'emissionprob_open_'):
            self.emissionprob_open_ = self.random_state.rand(
                4, self.n_features)
            normalize(self.emissionprob_open_, axis=1)
        if self._needs_init('e', 'emissionprob_closed_'):
            self.emissionprob_closed_ = self.random_state.rand(
                self.n_features)
            normalize(self.emissionprob_closed_, axis=0)

    def _check(self):
        super()._check()

        assert self.n_features is not None
        if self.emissionprob_open_.shape != (4, self.n_features):
            raise ValueError(
                f"emissionprob_open_ must have shape"
                f"({4}, {self.n_features})")
        self._check_sum_1("emissionprob_open_")
        if self.emissionprob_closed_.shape != (self.n_features,):
            raise ValueError(
                f"emissionprob_closed_ must have shape"
                f"({self.n_features},)")
        self._check_sum_1("emissionprob_closed_")

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if 'e' in self.params:
            DINUCLEOTIDES = ["TC", "GC", "CC", "AC"]
            self.emissionprob_open_ = np.empty((4, self.n_features))
            for i, dinucleotide in enumerate(DINUCLEOTIDES):
                self.emissionprob_open_[i] = np.maximum(
                    self.emissionprob_prior - 1 + stats[f'obs_{dinucleotide}'][1], 0)
            normalize(self.emissionprob_open_, axis=1)
            self.emissionprob_closed_ = np.maximum(
                self.emissionprob_prior - 1 + np.sum(stats[f'obs_{dinucleotide}'][0] for dinucleotide in DINUCLEOTIDES), 0)
            normalize(self.emissionprob_closed_, axis=0)
            # constrain editing rate in closed state to be low
            if self.emissionprob_closed_[1] > 0.01:
                self.emissionprob_closed_ = np.array([0.99, 0.01])

    def _check_and_set_n_features(self, X):
        """
        Check if ``X`` is a sample from a categorical distribution, i.e. an
        array of non-negative integers.
        """
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Symbols should be integers")
        if X.min() < 0:
            raise ValueError("Symbols should be nonnegative")
        if self.n_features is not None:
            if self.n_features - 1 < X.max():
                raise ValueError(
                    f"Largest symbol is {X.max()} but the model only emits "
                    f"symbols up to {self.n_features - 1}")
        else:
            self.n_features = X.max() + 1

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "e": nc * (nf - 1),
        }

    def _compute_likelihood(self, X):
        assert X.shape[1] == len(self.ref_seq)
        assert len(X.shape) == 2
        emissionprob_ = np.empty((len(self.ref_seq), self.n_features, self.n_components))
        for i in range(len(self.ref_seq)):
            if self.ref_seq[i] == "C":
                if i == 0:
                    emissionprob_[0, :, 0] = self.emissionprob_closed_
                    emissionprob_[0, :, 1] = self.emissionprob_open_.mean(axis=0)
                else:
                    emissionprob_[i, :, 0] = self.emissionprob_closed_
                    emissionprob_[i, :, 1] = self.emissionprob_open_["TGCA".index(self.ref_seq[i - 1])]
            else:
                emissionprob_[i, 0, :] = 1
                emissionprob_[i, 1, :] = 0
        return np.take_along_axis(emissionprob_[np.newaxis], X[..., np.newaxis, np.newaxis], axis=2).squeeze(axis=2).reshape(-1, self.n_components)

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        for dinucleotide in ["TC", "GC", "CC", "AC"]:
            stats[f'obs_{dinucleotide}'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(stats=stats, X=X,
                                                  lattice=lattice,
                                                  posteriors=posteriors,
                                                  fwdlattice=fwdlattice,
                                                  bwdlattice=bwdlattice)

        if 'e' in self.params:
            assert len(X.shape) == 2
            assert X.shape[0] * X.shape[1] == posteriors.shape[0]
            assert X.shape[1] == len(self.ref_seq)
            posteriors_ = posteriors.reshape(X.shape[0], X.shape[1], self.n_components)
            for dinucleotide, positions in self._sequence_contexts.items():
                np.add.at(stats[f'obs_{dinucleotide}'].T, X[:, positions].flatten(), posteriors_[:, positions].reshape(-1, self.n_components))

    def _generate_sample_from_state(self, state, sequence_context: str, random_state=None):
        if not sequence_context.endswith("C"):
            return [0]
        if state == 0:
            emissionprob_ = self.emissionprob_closed_
        elif state == 1:
            if len(sequence_context) >= 2:
                emissionprob_ = self.emissionprob_open_["TGCA".index(sequence_context[-2])]
            else:
                emissionprob_ = self.emissionprob_open_.mean(axis=0) # average over all contexts, might happen if C is at the start of the sequence
        else:
            raise ValueError("Invalid state")
        cdf = np.cumsum(emissionprob_)
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]

    def sample(self, n_samples=1, random_state=None, currstate=None):
        """
        Generate random samples from the model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        random_state : RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.
        currstate : int
            Current state, as the initial state of the samples.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Feature matrix.
        state_sequence : array, shape (n_samples, )
            State sequence produced by the model.

        Examples
        --------
        ::

            # generate samples continuously
            _, Z = model.sample(n_samples=10)
            X, Z = model.sample(n_samples=10, currstate=Z[-1])
        """
        check_is_fitted(self, "startprob_")
        self._check()

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        transmat_cdf = np.cumsum(self.transmat_, axis=1)

        if currstate is None:
            startprob_cdf = np.cumsum(self.startprob_)

        state_sequence = np.empty((n_samples, len(self.ref_seq)), dtype=int)
        X = np.empty((n_samples, len(self.ref_seq)), dtype=int)

        for i in range(n_samples):
            if currstate is None:
                state_sequence[i, 0] = (startprob_cdf > random_state.rand()).argmax()
            else:
                state_sequence[i, 0] = currstate
            X[i, 0] = self._generate_sample_from_state(
                currstate, self.ref_seq[0], random_state=random_state)[0]

            for t in range(1, len(self.ref_seq)):
                currstate = (
                    (transmat_cdf[currstate] > random_state.rand()).argmax())
                state_sequence[i, t] = currstate
                X[i, t] = self._generate_sample_from_state(
                    currstate, self.ref_seq[t-1:t+1], random_state=random_state)[0]

        return X, state_sequence
