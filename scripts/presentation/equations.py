"""Render display equations to transparent PNGs sized in slide inches.

matplotlib's mathtext (Computer Modern font set) is used rather than a LaTeX
installation so the deck rebuilds anywhere the project's virtualenv runs.

An equation rendered at ``pt`` points comes back with its natural width and
height in inches; placing the image at exactly that size makes the glyphs
appear on the slide at the requested point size, which is what keeps the maths
optically matched to the surrounding text.

Unsupported by mathtext, do not use: ``\\tfrac``, ``\\big``, ``\\le``,
``\\stackrel``. Use ``\\frac``, plain delimiters, ``\\leq``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from style import NAVY  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "figures" / "deck" / "eq"
DPI = 400


def render(tex: str, pt: float = 19.0, color: str = NAVY,
           cache_dir: Path | None = None) -> tuple[Path, float, float]:
    """Return ``(png_path, width_in, height_in)`` for a mathtext string.

    ``tex`` is given without the surrounding ``$``.
    """
    out_dir = cache_dir or CACHE
    out_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{tex}|{pt}|{color}|{DPI}".encode()).hexdigest()[:16]
    png = out_dir / f"eq_{key}.png"
    meta = png.with_suffix(".txt")
    if png.exists() and meta.exists():
        w, h = (float(v) for v in meta.read_text().split())
        return png, w, h

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=pt, color=color,
             math_fontfamily="cm", ha="left", va="bottom")
    fig.savefig(png, dpi=DPI, transparent=True, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)

    from PIL import Image
    with Image.open(png) as im:
        w, h = im.width / DPI, im.height / DPI
    meta.write_text(f"{w} {h}")
    return png, w, h


# --- the deck's equations, named ---------------------------------------------
# Kept in one place so a slide references an equation by name and the text of
# the equation is reviewable without reading layout code.
EQ = {
    # I. the machine
    "neuron": r"a(\rho)\;=\;\mathrm{Tr}\!\left[\;\varphi_T\!\left(H(\theta)\right)\,\rho\;\right],"
              r"\qquad H(\theta)=\sum_{j}\theta_j P_j",
    "hypclass": r"\mathcal{F}=\left\{\;\rho\;\longmapsto\;\mathrm{sign}\,\mathrm{Tr}(\rho H)"
                r"\quad:\quad H=\sum_j\theta_j P_j\;\right\}",
    "gibbs": r"\rho(\beta)\;=\;\frac{e^{-\beta H}}{\mathrm{Tr}\,e^{-\beta H}},"
             r"\qquad \beta=1/k_{\mathrm{B}}T",
    # II. the data structure
    "sector": r"\dim\mathcal{H}_{N,\,S_z}\;=\;\binom{n}{N_\alpha}\binom{n}{N_\beta}"
              r"\qquad\overset{n=8,\;N=8,\;S_z=0}{\longrightarrow}\qquad"
              r"\binom{8}{4}^{\!2}=4{,}900",
    "hamiltonian": r"H\;=\;E_{\mathrm{core}}\;+\;\sum_{pq,\sigma}h^{\mathrm{eff}}_{pq}\,"
                   r"a^{\dagger}_{p\sigma}a_{q\sigma}\;+\;\frac{1}{2}"
                   r"\sum_{pqrs,\sigma\tau}g_{pqrs}\,"
                   r"a^{\dagger}_{p\sigma}a^{\dagger}_{r\tau}a_{s\tau}a_{q\sigma}",
    "eigenblock": r"\rho(\beta)=\sum_{k=1}^{m}p_k\,|E_k\rangle\langle E_k|"
                  r"\;\;\equiv\;\;V^{\!\top}\,\mathrm{diag}(p)\,V,"
                  r"\qquad p_k=\frac{e^{-\beta(E_k-E_0)}}{Z}",
    "purification": r"|\Psi\rangle=\sum_{k=1}^{m}\sqrt{p_k}\;|E_k\rangle_{S}\otimes|k\rangle_{A},"
                    r"\qquad \mathrm{Tr}_{A}\,|\Psi\rangle\langle\Psi|=\rho",
    "mpsbound": r"\|\rho-\tilde{\rho}\|_{1}\;\leq\;2\,\left\|\,|\Psi\rangle-|\tilde{\Psi}\rangle\,\right\|_{2}"
                r"\;\leq\;2\sum_{j}\varepsilon_j",
    "tail": r"\mathrm{tail}\;\leq\;\sum_{i=m}^{k-1}e^{-(E_i-E_0)/k_{\mathrm{B}}T}"
            r"\;+\;(\dim-k)\,e^{-(E_{k-1}-E_0)/k_{\mathrm{B}}T}",
    # III. the label problem
    "features": r"x_m=\left(\mathrm{Tr}(\rho_m P_1),\;\ldots,\;\mathrm{Tr}(\rho_m P_{248})\right),"
                r"\qquad P_j\in\{\,Z_w,\;Z_iZ_j,\;X_iX_j,\;Y_iY_j\,\}",
    "dephasing": r"\mathrm{Tr}(\rho A)\;-\;\mathrm{Tr}\!\left(\Delta(\rho)A\right)"
                 r"\;=\;\sum_{i\neq j}\rho_{ij}A_{ji}\;=\;\mathrm{Tr}\!\left(\rho\,A_{\mathrm{od}}\right)",
    "stripdiag": r"y_m=\mathrm{sign}\left(\mathrm{Tr}\!\left(\rho_m A_{\mathrm{od}}\right)-\theta\right)",
    "spinsplit": r"S^{2}\;=\;D\;+\;S^{2}_{\mathrm{od}}\qquad\quad"
                 r"c_m\;=\;\mathrm{Tr}\!\left(\rho_m S^{2}_{\mathrm{od}}\right)"
                 r"\;=\;\langle S^{2}\rangle_m-\langle D\rangle_m",
    "screen": r"\mathrm{score}(y)\;=\;\frac{\left\|\,\mathrm{offdiag}(R_{+}-R_{-})\,\right\|_{F}}"
              r"{\left\|\,\mathrm{diag}(R_{+}-R_{-})\,\right\|_{F}},"
              r"\qquad R_{\pm}=\!\!\sum_{m\,:\;y_m=\pm1}\!\!\rho_m",
    # IV. depth
    "hybrid": r"a_i=\mathrm{Tr}\!\left[\varphi(B_i)\,\rho\right],\quad B_i=\sum_j\Theta_{ij}H_j"
              r"\qquad\longrightarrow\qquad \mathrm{MLP}\;\longrightarrow\;\hat{y}",
    "backprop": r"\frac{\partial\mathcal{L}}{\partial\Theta_{ij}}"
                r"\;=\;\mathrm{Tr}\!\left[\,H_j\;D\varphi(B_i)\left[R_i\right]\,\right],"
                r"\qquad R_i=\frac{1}{M}\sum_{m}\delta_{m,i}\,\rho_m",
    "dalecki": r"D\varphi(B)[X]=U\left(F\circ\left(U^{\dagger}XU\right)\right)U^{\dagger},"
               r"\qquad F_{k\ell}=\varphi^{[1]}(\lambda_k,\lambda_\ell)"
               r"=\frac{\varphi(\lambda_k)-\varphi(\lambda_\ell)}{\lambda_k-\lambda_\ell}",
    "commuting": r"\left[H_j,H_{j'}\right]=0\quad\Longrightarrow\quad"
                 r"a_i\ \ \mathrm{and}\ \ \frac{\partial\mathcal{L}}{\partial\Theta_{ij}}"
                 r"\ \ \mathrm{depend\ on}\ \rho\ \mathrm{only\ through}\ \mathrm{diag}(\rho)",
    # V. scale
    "qubits": r"Q=2n\ \ \mathrm{wires}\qquad\quad\dim=\binom{n}{N/2}^{\!2}"
              r"\qquad\quad \mathrm{conversion\ cost}\;\sim\;m\cdot 2^{Q}",
}


# --- the background talk on molecular Hamiltonians and thermal states -------
EQ.update({
    "electronic": r"H\;=\;-\frac{1}{2}\sum_{i}\nabla_i^{2}"
                  r"\;-\;\sum_{i,A}\frac{Z_A}{|\mathbf{r}_i-\mathbf{R}_A|}"
                  r"\;+\;\sum_{i<j}\frac{1}{|\mathbf{r}_i-\mathbf{r}_j|}",
    "second_quant": r"H\;=\;\sum_{pq,\sigma}h_{pq}\,a^{\dagger}_{p\sigma}a_{q\sigma}"
                    r"\;+\;\frac{1}{2}\sum_{pqrs,\sigma\tau}g_{pqrs}\,"
                    r"a^{\dagger}_{p\sigma}a^{\dagger}_{r\tau}a_{s\tau}a_{q\sigma}",
    "integrals": r"h_{pq}=\int\varphi_p(\mathbf{r})\,\hat{h}\,\varphi_q(\mathbf{r})\,d\mathbf{r},"
                 r"\qquad g_{pqrs}=\iint\frac{\varphi_p(\mathbf{r}_1)\varphi_q(\mathbf{r}_1)\,"
                 r"\varphi_r(\mathbf{r}_2)\varphi_s(\mathbf{r}_2)}"
                 r"{|\mathbf{r}_1-\mathbf{r}_2|}\,d\mathbf{r}_1 d\mathbf{r}_2",
    "scf": r"F(C)\,C\;=\;S\,C\,\varepsilon",
    "ao2mo": r"g_{pqrs}\;=\;\sum_{\mu\nu\lambda\sigma}"
             r"C_{\mu p}C_{\nu q}C_{\lambda r}C_{\sigma s}\;(\mu\nu|\lambda\sigma)",
    "dim_active": r"\dim\;=\;\binom{n}{N_\alpha}\binom{n}{N_\beta}"
                  r"\qquad\quad\mathrm{water,\ full\ basis:}\;\;"
                  r"\binom{24}{5}^{\!2}\;=\;1.8\times10^{9}",
    "frozen": r"h^{\mathrm{eff}}_{pq}=h_{pq}+\sum_{i\in\mathrm{core}}"
              r"\left[\,2(pq|ii)-(pi|iq)\,\right],"
              r"\qquad E_{\mathrm{core}}=E_{\mathrm{nuc}}+2\sum_i h_{ii}"
              r"+\sum_{ij}\left[\,2(ii|jj)-(ij|ji)\,\right]",
    "gibbs_def": r"\rho(\beta)\;=\;\frac{e^{-\beta H}}{\mathrm{Tr}\,e^{-\beta H}}"
                 r"\;=\;\sum_{k}p_k\,|E_k\rangle\langle E_k|,"
                 r"\qquad p_k=\frac{e^{-\beta(E_k-E_0)}}{Z}",
    "truncation": r"\rho\;\approx\;\sum_{k=1}^{m}p_k\,|E_k\rangle\langle E_k|,"
                  r"\qquad \mathrm{discarded\ weight}\;=\;1-\sum_{k=1}^{m}p_k\;\leq\;\epsilon",
    "storage_eq": r"\mathrm{store}\;(p,\,V)\;\;\mathrm{with}\;\;"
                  r"\rho=V^{\!\top}\mathrm{diag}(p)\,V"
                  r"\qquad m\!\times\!\dim\;\;\mathrm{instead\ of}\;\;\dim^{2}",
    "occupation": r"|n_0\,n_1\,\ldots\,n_{M-1}\rangle,\quad n_p\in\{0,1\}"
                  r"\qquad\longleftrightarrow\qquad M\ \mathrm{qubits}",
    "jw": r"a_p\;=\;\left(\,Z_0\,Z_1\cdots Z_{p-1}\right)\;"
          r"\frac{X_p+iY_p}{2},\qquad a^{\dagger}_p a_p=\frac{I-Z_p}{2}",
    "pauli_form": r"H\;=\;\sum_{j}c_j\,P_j,\qquad P_j\in\{I,X,Y,Z\}^{\otimes Q},"
                  r"\qquad \#\{j\}\;=\;O(n^{4})",
    "purif_def": r"|\Psi\rangle=\sum_{k}\sqrt{p_k}\,|E_k\rangle_{S}\otimes|k\rangle_{A},"
                 r"\qquad \mathrm{Tr}_{A}|\Psi\rangle\langle\Psi|=\rho(\beta)",
    "imag_time": r"|\Psi(\beta)\rangle\;\propto\;\left(e^{-\beta H/2}\otimes I\right)"
                 r"|\Psi(0)\rangle,\qquad |\Psi(0)\rangle=\frac{1}{\sqrt{d}}"
                 r"\sum_{k}|k\rangle_S\otimes|k\rangle_A",
    "sector_reduce": r"2^{Q}=65{,}536\qquad\longrightarrow\qquad"
                     r"\dim\mathcal{H}_{N,S_z}=4{,}900",
    "observable": r"\langle A\rangle_\beta\;=\;\mathrm{Tr}\!\left[\rho(\beta)A\right],"
                  r"\qquad F=-k_{\mathrm{B}}T\ln Z,"
                  r"\qquad S=-k_{\mathrm{B}}\mathrm{Tr}\!\left[\rho\ln\rho\right]",
})


# --- the fermionic algebra, for the background talk ------------------------
EQ.update({
    "anticommute": r"\{a_p,\,a^{\dagger}_q\}=\delta_{pq}\,I,"
                   r"\qquad \{a_p,\,a_q\}=\{a^{\dagger}_p,\,a^{\dagger}_q\}=0",
    "fock_build": r"|n_0\,n_1\ldots n_{M-1}\rangle\;=\;"
                  r"(a^{\dagger}_0)^{n_0}(a^{\dagger}_1)^{n_1}\cdots"
                  r"(a^{\dagger}_{M-1})^{n_{M-1}}\,|\mathrm{vac}\rangle",
    "creation_phase": r"a^{\dagger}_p\,|n\rangle\;=\;(1-n_p)\,(-1)^{\Pi_p}"
                      r"\;|\ldots,1_p,\ldots\rangle,\qquad"
                      r"\Pi_p\equiv\sum_{q<p}n_q",
    "number_ops": r"\hat{n}_p=a^{\dagger}_pa_p,\qquad \hat{N}=\sum_p \hat{n}_p,"
                  r"\qquad [H,\hat{N}]=[H,\hat{S}_z]=[H,\hat{S}^2]=0",
    "fock_dim": r"\mathcal{F}=\bigoplus_{N=0}^{M}\mathcal{H}_N,"
                r"\qquad \dim\mathcal{F}=2^{M},"
                r"\qquad \dim\mathcal{H}_N=\binom{M}{N}",
    "ci_expansion": r"|\Psi\rangle=\sum_{K}c_K\,|K\rangle,"
                    r"\qquad H_{KK'}=\langle K|H|K'\rangle",
    "sc_diag": r"\langle K|H|K\rangle=\sum_{i\in K}h_{ii}"
               r"+\frac{1}{2}\sum_{i,j\in K}\left[(ii|jj)-(ij|ji)\right]",
    "sc_single": r"\langle K|H|K^{p}_{q}\rangle=h_{pq}"
                 r"+\sum_{i\in K}\left[(pq|ii)-(pi|iq)\right]",
    "sc_double": r"\langle K|H|K^{pr}_{qs}\rangle=(pq|rs)-(ps|rq),"
                 r"\qquad\quad \langle K|H|K'\rangle=0"
                 r"\;\;\mathrm{beyond\ two\ replacements}",
    "spin_sector": r"\dim\mathcal{H}_{N,S_z}=\binom{n}{N_\alpha}\binom{n}{N_\beta}"
                   r"\qquad\quad\mathrm{CAS}(8,8):\;\;\binom{8}{4}^{\!2}=4{,}900",
    "partition": r"Z=\mathrm{Tr}\,e^{-\beta H},\qquad F=-k_{\mathrm{B}}T\ln Z,"
                 r"\qquad \langle A\rangle_\beta=\mathrm{Tr}\!\left[\rho(\beta)A\right]",
    "bk_weight": r"\mathrm{Jordan\!-\!Wigner}:\;O(M)\qquad"
                 r"\mathrm{Bravyi\!-\!Kitaev}:\;O(\log M)",
    "hubbard": r"H=-t\sum_{\langle ij\rangle,\sigma}"
               r"a^{\dagger}_{i\sigma}a_{j\sigma}+U\sum_i \hat{n}_{i\uparrow}\hat{n}_{i\downarrow}",
    "yields": r"E(\mathbf{R}),\quad \nabla_{\mathbf{R}}E,\quad \{E_k\},"
              r"\quad \langle A\rangle_\beta,\quad F(T),\quad "
              r"\chi(\omega)",
})


def render_all(pt: float = 19.0) -> dict[str, tuple[Path, float, float]]:
    return {k: render(v, pt=pt) for k, v in EQ.items()}


if __name__ == "__main__":
    for name, (path, w, h) in render_all().items():
        print(f"{name:14s} {w:5.2f} x {h:4.2f} in   {path}")
