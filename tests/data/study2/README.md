# Polarisation of the $ZZ$ pair in the four-lepton final state

Working notes for case study 2 (`tests/data/study2`). They record the physics motivation, the simulation
choices and the observables worth developing. These are notes, not a publication draft.

---

## 1. Motivation

The longitudinal polarisation states of the $W$ and $Z$ bosons are the Goldstone modes provided by
electroweak symmetry breaking, so their couplings are the most direct handle on the mechanism itself.
Measuring the polarisation of a produced gauge boson probes more than its momentum spectrum does.

Two properties of $ZZ$ production make it a suitable case.

First, the polarisation states are not equally populated, and their relative weight is a calculable
prediction that depends on the production mechanism. For the dominant quark-initiated process
$q\bar q\to ZZ$, the longitudinal component is strongly suppressed relative to the transverse one at high
four-lepton invariant mass. The $Z_L$ coupling to a light initial-state quark is proportional to
$m_q/m_Z$, so the longitudinal rate is fed by the small heavy-flavour and gluon content of the proton
rather than by the valence quarks. The longitudinal fraction therefore decreases with $m_{4\ell}$, and
measuring that dependence tests the Standard Model [1]. The complementary case, an amplitude that grows
with energy and is unitarised by Higgs exchange, belongs to electroweak $ZZjj$ production. That is why
the diboson polarisation programme has an inclusive and a vector-boson-fusion branch.

Second, the polarised cross sections are not separately observable. What is observable is the angular
distribution of the four leptons, and the polarisation amplitudes interfere. Only the sum is a physical
rate. Extracting polarisation fractions is therefore a template fit. One builds Monte Carlo templates for
$Z_LZ_L$, $Z_TZ_L$ and $Z_TZ_T$, adds an interference template, and fits their normalisations to data
[1,2,3].

In this study the longitudinally polarised sample $Z_LZ_L$ is the signal, and the transverse combinations
$Z_TZ_T$ and $Z_TZ_L$ are the background. The separation is not kinematic. All three share the same final
state and the same diagrams, and differ only through the spin structure of the angular distributions. A
fourth, unpolarised sample provides the reference rate from which the interference is obtained.

### 1.1 The interference term

The experimental definition is

$$
\sigma_{\rm int} = \sigma_{\rm incl} - \left(\sigma_{LL}+\sigma_{TL}+\sigma_{TT}\right),
$$

so the interference is obtained by subtraction, and it can be negative. In amplitude language,
$r_L = |\mathcal{M}_L|^2/|\mathcal{M}|^2$, $r_T = |\mathcal{M}_T|^2/|\mathcal{M}|^2$ and
$r_{\rm int} = 1-r_L-r_T$. Because the amplitudes are complex, $r_{\rm int}$ may be negative and $r_L$,
$r_T$ may individually exceed unity [4].

One expectation for this study. Both $Z$ bosons are generated exactly on shell, where the three physical
polarisations form a complete set, so $\sigma_{LL}+\sigma_{TL}+\sigma_{TT}$ should reproduce
$\sigma_{\rm incl}$ and the subtracted interference should be consistent with zero up to statistics. A
non-zero interference requires off-shell bosons, non-resonant contributions or higher orders. Here
off-shellness enters only through the MadSpin Breit-Wigner sampling. The check is worth doing explicitly,
because a non-zero inclusive-minus-polarised difference at generation level would mean that the
polarisation truncation is not behaving as documented.

---

## 2. The four samples

All samples are produced at leading order in QCD with `sm-full` in MadGraph5_aMC@NLO, so $gg\to ZZ$ is
absent by construction. It is loop induced and cannot be generated at tree level. The quark-induced
$q\bar q\to ZZ$ process is generated with all tree-level diagrams, including $q\bar q\to h\to ZZ$.

| sample | process | role |
|---|---|---|
| `zz_incl_4L` | `p p > z z` | unpolarised reference, enters through the interference subtraction |
| `zz_ll_4L` | `p p > z{0} z{0}` | signal, both $Z$ longitudinal |
| `zz_lt_4L` | `p p > z{0} z{T}` | background, one longitudinal one transverse |
| `zz_tt_4L` | `p p > z{T} z{T}` | background, both transverse, the largest component |

`{0}` selects the longitudinal polarisation and `{T}` the transverse one, both helicities summed. The
quantisation axis is the partonic centre-of-mass frame, which for a $2\to2$ process coincides with the
$ZZ$ rest frame. Both $Z$ bosons decay leptonically through MadSpin in the density mode
(`spinmode full`). In this mode the production polarisation is retained through the decay and the decay
amplitude is evaluated with off-shell matrix elements.

Because the $h\to ZZ$ diagrams are retained, the $Z_LZ_L$ sample contains the 125 GeV Higgs peak while
the transverse samples do not. The Higgs couples only to longitudinal $Z$ bosons. This is visible in
$m_{4\ell}$ and is one reason to treat $m_{4\ell}$ as a separate variable rather than as a training
input (Section 5).

The $Z$ bosons are on shell at generation, so the polarisation is defined unambiguously and the off-shell
treatment enters only in the decay. Each sample contains 1000 events, which is enough to exercise the
toolkit and not to produce physics results.

---

## 3. Which final states to study separately

The $ZZ\to\ell^+\ell^-\ell'^+\ell'^-$ final state splits into three experimentally distinct channels,
which should be studied separately and combined only at the level of the extracted fractions.

**$2e2\mu$, the mixed-flavour channel.** The cleanest channel for polarisation work and the natural
development target. The two $Z$ candidates are identified by flavour, the electron pair is one $Z$ and
the muon pair the other, so there is no pairing ambiguity, no combinatorial choice to optimise and no
ambiguity in the angular variables. It also has twice the yield of either same-flavour channel. With
$\mathrm{BR}(Z\to\ell\ell)\approx3.36\%$ per lepton flavour one finds
$\mathrm{BR}(2e2\mu)\approx2\,\mathrm{BR}(4e)$. Its drawback is that the two $Z$ slots have different
detector response depending on which one is the electron pair, so under a kinematic ordering rule the
first $Z$ is sometimes an electron pair and sometimes a muon pair. Either accept that mixture or record
the flavour configuration as a feature.

**$4\mu$.** Best dimuon mass resolution and lowest instrumental background, hence the most precise
angular variables and the most sensitive channel for the shape of $\cos\theta_i$. Its complication is
that both $Z$ bosons are the same particle and both decay to the same flavour, so nothing in the event
tags which muon pair came from which $Z$. The four muons can be grouped into two opposite-charge
same-flavour (OSSF) pairs in two ways, and a pairing rule is required (Section 5.1). It has the lowest
yield of the three.

**$4e$.** Largest backgrounds and the most delicate systematics. Final-state radiation and photon
recombination shift the dilepton masses, and electron charge misidentification produces a background with
wrong-sign pairs. The same pairing ambiguity as $4\mu$ applies. It should not be
merged with $4\mu$, because merging would hide the resolution and reconstruction effects that dominate
the angular measurement.

**Identical particles.** In the same-flavour channels the two $Z$ bosons are the same particle and the
two lepton pairs are indistinguishable, which has a consequence beyond the reconstruction choice just
described. The amplitude for $pp\to\ell^+\ell^-\ell^+\ell^-$ receives contributions from two resonance
structures, one in which the first $Z$ couples to the pair $(\ell_1\ell_2)$ and the second to
$(\ell_3\ell_4)$, and one in which the roles are crossed. The two route different momenta through the two
$Z$ propagators, nothing in the final state distinguishes them, and they interfere. Amplitudes are
therefore summed before squaring, so the two pairings are not independent rates to be added. In $2e2\mu$
the crossed structure is absent, because it would require a lepton-flavour-violating $Z$ coupling. In the
samples used here it is absent as well. The $Z$ bosons are on shell and the decay is treated in the
narrow-width approximation, so only the pairing near the pole survives, and the crossed term belongs to
the non-resonant contributions covered by the signal-definition caveat in Section 6. The pairing rule of
Section 5.1 is therefore a reconstruction convention and not a physical assignment, and no result should
depend on which of the two allowed pairings was chosen.

**Recommendation.** Develop and validate the full chain on $2e2\mu$, then train and evaluate the
discriminator separately in $4\mu$ and $4e$, and combine the three only when extracting the fractions,
with channel-specific templates. A single classifier trained across channels is acceptable only if the
channel is provided as an input. Otherwise the network learns flavour-dependent reconstruction artefacts
rather than spin correlations.

---

## 4. Observables

### 4.1 Frame and ordering conventions

Everything below has to be defined once and applied identically to all samples.

- The $ZZ$ system is boosted to its rest frame, the four-lepton centre-of-mass frame, in which the two
  $Z$ bosons are back to back.
- Each $Z$ is then boosted to its own rest frame to define the lepton angles.
- The two $Z$ candidates are ordered by a fixed rule. ATLAS orders them by the largest absolute dilepton
  rapidity, $|y_{\ell\ell}|$ [1]. Ordering by $p_T$ is equally defensible but must be fixed, because
  $\cos\theta_1$ and $\cos\theta_2$ are not symmetric under exchange and the sign of the azimuthal
  variables changes with the choice.
- Leptons are dressed, with photons recombined inside a small cone, and the four leptons are the ones
  forming the two $Z$ candidates.

### 4.2 The polarisation-sensitive angular variables

These carry essentially all of the spin information, and they are the variables used by ATLAS in the
published observation [1]. They are also what a cut-based analysis should start from.

| variable | definition | why it discriminates |
|---|---|---|
| $\cos\theta_1$, $\cos\theta_2$ | angle between the negatively charged lepton, in the rest frame of $Z_i$, and the direction of flight of $Z_i$ in the four-lepton rest frame | a longitudinal $Z$ decays as $\sin^2\theta$ and a transverse one as $(1+\cos^2\theta)$, so $\lvert\cos\theta_i\rvert$ peaks at 0 for $L$ and near 1 for $T$ |
| $\Delta\phi_{\ell_1\ell_2}$, $\Delta\phi_{\ell_3\ell_4}$ | azimuthal opening angle of the two leptons of each $Z$, in the four-lepton rest frame | encodes the spin correlation between the two bosons, not only their individual polarisations |
| $\cos\theta^*_{Z_1}$ | production angle of the leading $Z$, in the four-lepton rest frame | separates the production kinematics of the $LL$ and $TT$ amplitudes, which differ in their $t$-channel structure |

The distribution in $\cos\theta_i$ alone is already a usable discriminant, and the azimuthal variables add
the interference-sensitive information. A cut-based analysis naturally starts from a two-dimensional
selection in $(|\cos\theta_1|,|\cos\theta_2|)$, optionally with a cut on $\Delta\phi$, followed by a
template fit rather than a hard signal/background split. A hard cut discards the shape information the
measurement is built on.

### 4.3 Kinematic and event-level variables

| group | variables | comment |
|---|---|---|
| diboson system | $m_{4\ell}$, $p_T^{4\ell}$, $\lvert y_{4\ell}\rvert$ | $m_{4\ell}$ carries the Higgs peak and the $LL$-suppression trend. It is a fit variable, and feeding it to a classifier lets the classifier use the mass spectrum instead of the spin structure |
| individual bosons | $p_T(Z_1)$, $p_T(Z_2)$, $m(Z_1)$, $m(Z_2)$, $\Delta\phi(Z_1,Z_2)$, $\Delta y(Z_1,Z_2)$, $\Delta R(Z_1,Z_2)$ | $m(Z_i)$ encodes off-shellness, and the separation between the two $Z$ candidates is correlated with the production mechanism |
| global shape | aplanarity, sphericity, $H_T$, thrust | summarise the radiation pattern. Useful in a real analysis, of limited use here where all samples share the same production |
| activity | $N_{\rm jets}$, $p_{T}^{\rm jet}$, $\lvert\eta^{\rm jet}\rvert$ | separates electroweak $qq\to qqZZ$ from inclusive $q\bar q\to ZZ$. Not needed for this study, essential as soon as VBF enters |
| lepton quality | lepton $p_T$ ordering, isolation, impact-parameter significances, charge | the handles on the reducible background in a real analysis |

Two inputs are excluded from the training set. The first is lepton flavour or
configuration, which leaks reconstruction differences between channels. The second is the generator-level
polarisation label, which is the target itself.

---

## 5. Analysis strategy

### 5.1 Reconstruction prerequisites

The reconstruction has to be fixed before any multivariate work.

1. Build all OSSF pairs with $m_{\ell\ell}>5$ GeV.
2. Form the quadruplet from two such pairs. For $4e$ and $4\mu$ choose the pairing that minimises
   $\sum_i |m_{\ell\ell,i}-m_Z|$. For $2e2\mu$ the pairing is unique.
3. Require $|m_{\ell\ell}-m_Z|<\Delta$ for both pairs with $\Delta\approx10$ GeV, and $m_{4\ell}>180$ GeV.
   The ATLAS signal region uses a 10 GeV mass window and is then split at $m_{4\ell}=240$ GeV [1].
4. Order the two $Z$ candidates by the chosen rule and compute the angles.

The selection is itself a statement about polarisation. Requiring both $Z$ candidates near the pole
discards the off-shell configurations in which the polarised decomposition is least well defined. That is
the right choice for a fraction measurement, and it is why the interference term is small inside the
fiducial region, but it means this study does not address the fully off-shell case.

### 5.2 Cut-based analysis

A fiducial selection, then a window in $|\cos\theta_1|$ and $|\cos\theta_2|$ chosen to enhance $LL$, then
a template fit in a discriminating distribution. It is interpretable, statistically robust at 1000 events
and the natural cross-check of the multivariate result. Its weakness is that the correlation between
$\cos\theta_i$ and $\cos\theta^*_{Z_1}$ is not exploited.

### 5.3 Boosted decision tree

The ATLAS analysis trains a BDT to separate $Z_LZ_L$ from $Z_TZ_X$ using the five angular observables of
Section 4.2 [1]. That is the baseline adopted here. The input set is small, all of it with a clear
physical meaning, and it is trained separately per channel with the samples split into training and
validation subsets.

Some points of training hygiene are worth stating.

- Train on the same generator configuration for signal and background, otherwise the network finds
  generator artefacts rather than spin structure.
- Report the ROC or AUC for bookkeeping, but judge the model on the bias and the uncertainty of the
  fitted $LL$ fraction in pseudo-experiments, and on a closure test in which the three polarised
  templates plus the interference template reproduce the inclusive distribution.
- Check the stability of the output distribution under the choice of the $Z$ ordering rule and under the
  $Z$ mass window. A discriminator that collapses when the ordering changes is learning the ordering,
  not the polarisation.
- With 1000 events per sample, use a shallow ensemble and few inputs. There are five angular variables,
  so the capacity required is genuinely small and a deep network will memorise the training sample.

### 5.4 Amplitude-assisted discriminators

An alternative with a clear physics motivation is to use the matrix element as the discriminant. One
computes $r_L$ and $r_T$ per event and uses the likelihood ratio $r_L/(r_L+r_T)$ as the tagger, or feeds
the per-event amplitude ratios to a network. This is close to optimal because it uses the fully
differential information rather than a few projections [4], and it has been demonstrated in $Z$+jet and
in $ZZ$ scattering with deep networks [4,5]. For a toolkit example it has the practical advantage that
per-event polarisation weights can be stored in the ntuple, so the classifier becomes a branch rather
than a trained model.

---

## 6. Known pitfalls

- **Frame and ordering conventions.** A polarisation measurement is meaningful only relative to a stated
  frame and a stated ordering. Two analyses that differ in either will disagree before any physics enters.
- **Fractions are not bounded.** Once interference is separated out, $r_{\rm int}$ may be negative and
  $r_L$, $r_T$ may exceed unity [4]. Fitted fractions must be allowed to be negative, and any constraint
  to the physical region must be stated.
- **$gg\to ZZ$.** Loop induced and therefore absent from these samples. Experimentally it is a
  non-trivial component whose polarised decomposition is generator dependent, and ATLAS treats it
  inclusively as a background because reweighted polarised predictions were found to disagree with
  fixed-order calculations [1]. Any extension beyond leading order has to confront it.
- **Signal definition.** Polarised cross sections depend on whether the narrow-width approximation, the
  pole approximation or the full off-shell calculation is used, and the three differ at the few per cent
  level in the fiducial region and more in the tails [2]. The definition must be stated.
- **Interference as a difference.** Subtracting two Monte Carlo predictions propagates the full
  statistical error of both. With 1000 events the subtracted term is noise dominated. Constructing the
  interference by reweighting the same events, as in [1], is far superior statistically and is the
  natural next step here.
- **Statistics.** 1000 events per sample is enough to exercise a pipeline and to see the shape of the
  angular distributions, and nowhere near enough to fit three fractions. Any number quoted from these
  samples is a sanity check, not a result.

---

## 7. Role as a `hepdataset` example

The parent repository converts Delphes output into analysis-ready trees. This study serves as the
worked example that exercises the parts of that chain a single-process example cannot reach.

- **Four labelled samples with identical reconstruction.** The same code has to process four
  physics-identical final states that differ only in a quantum-number label. That is the simplest
  realistic demonstration of sample bookkeeping, of a shared selection and of per-sample metadata
  carried through to the ntuple.
- **A genuine reconstruction problem.** Building two $Z$ candidates from four leptons, resolving the
  pairing ambiguity in the same-flavour channels and ordering the candidates is the kind of multi-object
  reconstruction the object-selection and kinematics layers exist to support.
- **Frame-dependent variables.** The decay angles require successive boosts into the four-lepton and
  $Z$ rest frames, which tests the kinematics module beyond invariant masses and $\Delta R$.
- **Three channels of one final state.** $2e2\mu$, $4\mu$ and $4e$ exercise channel-dependent handling.
  This includes identical-particle combinatorics, dressing, per-channel selections, and the option of a
  combined run.
- **A target that is not a cut.** The physics goal is a fit to template normalisations, so the example
  motivates storing per-event weights and systematic variations rather than only a selected tree.

The samples are small. Their purpose is to make the full chain, from generator through
decay, shower, detector, reconstruction and ntuple to classifier, runnable and inspectable end to end in
minutes, not to support a measurement.

---

## 8. References

A note on the literature. The inclusive $ZZ\to4\ell$ polarisation measurement is, at the time of writing,
an ATLAS result [1], while the CMS polarisation programme has concentrated on $WZ$, on same-sign $WW$ in
vector-boson scattering and on the VBS channels [8]. The ATLAS analysis is therefore the primary template
for this study, and the phenomenological comparisons [2,3,4] supply the definition-level caveats.

1. ATLAS Collaboration, *Observation of pair production of longitudinally polarized $Z$ bosons in
   four-lepton final state with the ATLAS detector*, [arXiv:2607.12879](https://arxiv.org/abs/2607.12879)
   (STDM-2024-01), combined with the Run 2 measurement, JHEP 12 (2023) 107. The BDT inputs, the $Z$
   ordering rule, the signal-region definition and the interference-by-subtraction prescription used in
   Sections 4 and 5 follow this analysis.
2. C. Carrivale *et al.*, *Precise Standard-Model predictions for polarised $Z$-boson pair production and
   decay at the LHC*, [arXiv:2505.09686](https://arxiv.org/abs/2505.09686). Comparison of signal
   definitions and of non-resonant and interference effects between generators.
3. A. Denner and G. Pelliccioli, *NLO EW and QCD corrections to polarized $ZZ$ production in the
   four-charged-lepton channel at the LHC*, JHEP 10 (2021) 097,
   [arXiv:2107.06579](https://arxiv.org/abs/2107.06579).
4. *Amplitude-assisted tagging of longitudinally polarised bosons using wide neural networks*,
   [CDS record](https://cds.cern.ch/record/2868888). Defines $r_L$, $r_T$, $r_{\rm int}=1-r_L-r_T$ and the
   event-by-event likelihood-ratio tagger.
5. J. Lee *et al.*, *Polarization fraction measurement in $ZZ$ scattering using deep learning*,
   Phys. Rev. D 99 (2019) 033004, [arXiv:1812.07591](https://arxiv.org/abs/1812.07591).
6. D. Buarque Franzosi *et al.*, *Automated predictions from polarized matrix elements*, JHEP 04 (2020)
   082, [arXiv:1912.01725](https://arxiv.org/abs/1912.01725). The polarisation syntax used in the process
   definitions, and the frame dependence of the polarisation basis.
7. ATLAS Collaboration, *Observation of gauge boson joint-polarisation states in $WZ$ production*,
   [arXiv:2211.09435](https://arxiv.org/abs/2211.09435), and *Evidence for longitudinally polarised $Z$
   bosons in electroweak $WZ$ production in association with two jets*,
   [arXiv:2607.20246](https://arxiv.org/abs/2607.20246).
8. CMS Collaboration, *Measurements of production cross sections of polarized same-sign $WW$ boson pairs
   in association with two jets*, Phys. Lett. B 812 (2021) 136018,
   [arXiv:2009.09429](https://arxiv.org/abs/2009.09429).
