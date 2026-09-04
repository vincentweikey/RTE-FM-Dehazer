# RTE-FM-Dehazer

** RTE-FM-Dehazer: Radiative Transfer Equation Inspired Flow Matching for Real-World Image Dehazing**


Official PyTorch implementation of **RTE-FM-Dehazer**, accepted to **ECCV 2026**.

Single-image dehazing is typically cast as image-to-image translation, which
depends heavily on the Atmospheric Scattering Model (ASM) — whose assumptions of
single scattering and homogeneous media are often violated in real scenes,
causing residual haze and color drift. RTE-FM-Dehazer replaces the ASM prior
with the **Radiative Transfer Equation (RTE)**, which jointly models scattering
and absorption in non-homogeneous, multiple-scattering media.

Motivated by the structural similarity between the RTE diffusion–absorption
term and the probability-flow ODE, we introduce a **diffusion–absorption
regularizer** derived from a reduced RTE that steers the rectified-flow-matching
trajectory at every integration step. The method operates entirely in the
latent space of the Stable Diffusion VAE. Together with the method we release
**P-HAZE**, a dataset of **50,000 realistic hazy/clear pairs** built with an
automated vision-language-model pipeline. Trained solely on P-HAZE, RTE-FM-Dehazer
achieves leading results on five real-world dehazing benchmarks with strong
cross-domain generalization.


