Below is a **ready‑to‑paste** Markdown guide, `using_bede.md`.  It distils the official Bede documentation and complementary N8 CIR material into a single, self‑contained reference covering registration, access, hardware, storage, Slurm job submission (including Grace‑Hopper pilot nodes), software environments, and best‑practice tips.  Follow the step‑by‑step sections or use the hyperlinked table of contents to jump straight to what you need.

---

## Table of Contents

1. [What Is Bede?](#what-is-bede)
2. [Getting Access](#getting-access)
3. [Logging‑In & Multi‑Factor Authentication](#logging-in--multi-factor-authentication)
4. [Graphical Access with X2Go](#graphical-access-with-x2go)
5. [Storage Layout & Quotas](#storage-layout--quotas)
6. [Node Architectures & Partitions](#node-architectures--partitions)
7. [Running Jobs with Slurm](#running-jobs-with-slurm)
8. [Grace‑Hopper Pilot (ARM + H100) Quick‑Start](#grace-hopper-pilot-arm--h100-quick-start)
9. [Software & Module Management](#software--module-management)
10. [Containers on Bede (Apptainer & Singularity)](#containers-on-bede-apptainer--singularity)
11. [MPI on Bede](#mpi-on-bede)
12. [Acknowledging Bede & Good Citizenship](#acknowledging-bede--good-citizenship)
13. [Troubleshooting & Support Routes](#troubleshooting--support-routes)

---

## What Is Bede? <a name="what-is-bede"></a>

* **Bede** is the N8 universities’ Tier‑2 GPU‑accelerated supercomputer hosted at Durham University.  The flagship partition comprises 32 IBM Power9 dual‑socket nodes, each with four NVIDIA V100 GPUs connected by NVLink and dual‑rail EDR InfiniBand (100 Gb s‑¹) interconnect, mirroring the architecture of the U.S. Summit and Sierra systems([n8cir.org.uk][1]).
* Since 2024 Bede also offers an **open pilot of six NVIDIA Grace‑Hopper (GH200) “superchips”**—a tightly coupled 72‑core Arm CPU and 96 GB H100 GPU sharing 900 GB s‑¹ of NVLink‑C2C bandwidth([bede-documentation.readthedocs.io][2]).
* A 2 PB Lustre filesystem delivers ≈ 10 GB s‑¹ sustained I/O, complemented by modest NFS home/project areas([bede-documentation.readthedocs.io][3]).

---

## Getting Access <a name="getting-access"></a>

1. **Project‑based accounts.** Access is granted per research project; create or join a project via the EPCC SAFE portal.

   * Register the project on the N8 CIR portal first → then create a SAFE account and select *Project → Request access*([bede-documentation.readthedocs.io][2]).
2. **Manual approval.** Sysadmins manually provision logins; you’ll receive confirmation when ready([bede-documentation.readthedocs.io][2]).
3. **Eligibility.** Researchers at any N8 institution qualify automatically; others may apply under EPSRC access routes([n8cir.org.uk][1]).

---

## Logging‑In & Multi‑Factor Authentication <a name="logging-in--multi-factor-authentication"></a>

| Method            | Hostname         | Notes                                                                                    |
| ----------------- | ---------------- | ---------------------------------------------------------------------------------------- |
| SSH (recommended) | `bede.dur.ac.uk` | Front‑end to `login1` & `login2` (Power9)([bede-documentation.readthedocs.io][2])        |
| X2Go              | same hostname    | First login **must** be SSH so you can enrol MFA([bede-documentation.readthedocs.io][2]) |

* **MFA flow**: after first SSH login you (i) change your password, (ii) scan the TOTP QR code using an authenticator app, then log in with *password* (First Factor) + *6‑digit TOTP* (Second Factor) thereafter([bede-documentation.readthedocs.io][2]).
* **Maintenance window**: every **Tuesday 08:00–10:00 UK time** the login nodes may reboot; batch jobs run uninterrupted([bede-documentation.readthedocs.io][2]).

---

## Graphical Access with X2Go <a name="graphical-access-with-x2go"></a>

X2Go tunnels a lightweight desktop for GUI tools (e.g. MATLAB, ParaView):

1. Install the client from [https://wiki.x2go.org](https://wiki.x2go.org).  Packages exist for Windows, macOS DMG and most Linux distros([wiki.x2go.org][4], [wiki.x2go.org][5]).
2. Create a *Published Applications* session:

   * **Host** =`bede.dur.ac.uk`; **Login** =`<your username>`; **Session type** =`Published applications`([bede-documentation.readthedocs.io][2]).
3. After connection, click the blue *circle* icon, then **System → MATE Terminal** to launch a graphics‑enabled shell on the login node([bede-documentation.readthedocs.io][2]).
4. Use the *pause* icon to disconnect without closing apps; ensure you reconnect to the **same** login node by specifying `login1` or `login2` directly if you persist sessions([bede-documentation.readthedocs.io][2]).

---

## Storage Layout & Quotas <a name="storage-layout--quotas"></a>

| Path                          | Default Quota | Back‑up | Intended Use                                                                |
| ----------------------------- | ------------- | ------- | --------------------------------------------------------------------------- |
| `/users/$USER`                | 20 GB         | ✔       | Dotfiles, small scripts([bede-documentation.readthedocs.io][2])             |
| `/projects/$PROJECT`          | 20 GB         | ✔       | Shared project code/data([bede-documentation.readthedocs.io][2])            |
| `/nobackup/projects/$PROJECT` | none          | ✘       | Bulk, high‑throughput I/O on Lustre([bede-documentation.readthedocs.io][2]) |

Use `quota`, `df -h /projects/<project>`, or `du -csh /nobackup/projects/<project>` to monitor usage.  **Avoid personal data**; the facility is for research only([bede-documentation.readthedocs.io][2]).

---

## Node Architectures & Partitions <a name="node-architectures--partitions"></a>

| Partition | CPU Arch                 | GPU Model     | Typical Use            | Default / Max Walltime                                   |
| --------- | ------------------------ | ------------- | ---------------------- | -------------------------------------------------------- |
| `gpu`     | Power9 (`ppc64le`)       | 4× V100       | production GPU jobs    |  1 h / 2 d([bede-documentation.readthedocs.io][2])       |
| `infer`   | Power9                   | 4× T4         | low‑power inference    |  1 h / 2 d([bede-documentation.readthedocs.io][2])       |
| `test`    | Power9                   | 4× V100       | compile/tests ≤ 30 min |  15 min / 30 min([bede-documentation.readthedocs.io][2]) |
| `ghlogin` | Grace‑Hopper (`aarch64`) | H100 (shared) | interactive ARM dev    |  1 h / 8 h([bede-documentation.readthedocs.io][2])       |
| `gh`      | Grace‑Hopper             | 1× H100       | production ARM jobs    |  1 h / 2 d([bede-documentation.readthedocs.io][2])       |
| `ghtest`  | Grace‑Hopper             | 1× H100       | quick tests            |  15 min / 30 min([bede-documentation.readthedocs.io][2]) |

---

## Running Jobs with Slurm <a name="running-jobs-with-slurm"></a>

### 1. Single‑Node GPU job (Power9 example)

```bash
#!/bin/bash
#SBATCH --account=<project>
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1         # 1 × V100
#SBATCH --nodes=1

module load cuda
nvidia-smi
python my_script.py
```

This grants 1 GPU plus 25 % of the node’s CPUs/RAM([bede-documentation.readthedocs.io][2]).

### 2. Grace‑Hopper quick test

```bash
#!/bin/bash
#SBATCH --account=<project>
#SBATCH --partition=ghtest
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1   # full GH200 node
ghrun python3 benchmark.py      # or sbatch inside ghlogin
```

([bede-documentation.readthedocs.io][2])

### 3. Multi‑Node MPI (Power9)

```bash
#!/bin/bash
#SBATCH --account=<project>
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00

module load gcc cuda openmpi
bede-mpirun --bede-par 1ppg ./my_mpi_app
```

([bede-documentation.readthedocs.io][2])

`bede-mpirun` simplifies rank/thread placement; options include `1ppn`, `1ppg`, `1ppc`, or `1ppt`([bede-documentation.readthedocs.io][2]).

> **Tip:** Small test jobs (< 30 min, ≤ 2 nodes) receive higher priority in `test` or `ghtest` partitions([bede-documentation.readthedocs.io][2]).

---

## Grace‑Hopper Pilot (ARM + H100) Quick‑Start <a name="grace-hopper-pilot-arm--h100-quick-start"></a>

1. **Interactive shell**:

   ```bash
   ghlogin -A <project>               # 4 CPU, 16 GB, 8 h
   ghlogin -A <project> -c 8 --mem 24G --time 04:00:00
   ```

([bede-documentation.readthedocs.io][2])
2\. **Batch submission**:
*From standard login node* → `ghbatch my.slurm`, or inside `ghlogin` run `sbatch`/`srun` normally([bede-documentation.readthedocs.io][2]).
3\. **Software**: use `module avail` on `ghlogin` to list ARM builds.  Key differences:

* CUDA ≥ 11.8 targeting *compute 90*; CUDA 11.7 must embed PTX 80.
* Apptainer replaces Singularity and supports `--fakeroot`.
* PyTorch: pip wheels ≥ 2.4.0 for CUDA 12.4, or NGC containers.
* OpenMPI with CUDA enabled; MVAPICH2‑GDR not yet available([bede-documentation.readthedocs.io][2]).

4. **Hybrid dotfiles**: detect CPU arch in `.bashrc` with `arch=$(uname -i)` and branch accordingly([bede-documentation.readthedocs.io][2]).

---

## Software & Module Management <a name="software--module-management"></a>

* **Environment Modules (Lmod)** provide compiler, MPI, CUDA, AI frameworks, and domain software packages; run `module avail` or search per category within the *Software on Bede* tree([bede-documentation.readthedocs.io][6]).
* Compile codes with the matching architecture flags (`-mcpu=power9`, `-march=armv9-a`, etc.) and appropriate GPU *sm* target (`sm_70`, `sm_90`, …) to avoid runtime errors([bede-documentation.readthedocs.io][3]).

---

## Containers on Bede (Apptainer & Singularity) <a name="containers-on-bede-apptainer--singularity"></a>

| Architecture             | Default Tool       | Rootless Builds                                               | Notes                                                                                                                      |
| ------------------------ | ------------------ | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `aarch64` (Grace‑Hopper) | **Apptainer**      | ✔ `--fakeroot`                                                | Large images: set `APPTAINER_CACHEDIR=/nobackup/...` to protect quotas([bede-documentation.readthedocs.io][7])             |
| `ppc64le` (Power9)       | **Singularity CE** | root access required for image build; run containers normally | Containers are **not** cross‑arch portable—build on the same ISA you intend to run([bede-documentation.readthedocs.io][7]) |

Apptainer inherits Singularity syntax; the `singularity` CLI remains an alias but warns on `SINGULARITY_` env vars([bede-documentation.readthedocs.io][7]).

---

## MPI on Bede <a name="mpi-on-bede"></a>

* **OpenMPI** is the default on all partitions; CUDA‑aware if `cuda` module is also loaded.
* **MVAPICH2** (and high‑performance **mvapich2‑gdr** on Power9) provide superior GPU–GPU collectives and MPI‑3 RMA support — use when your application relies on asynchronous GPU communications([bede-documentation.readthedocs.io][8]).
* Compilers are abstracted: `mpicc`, `mpicxx`, `mpif90` wrappers work regardless of backend([bede-documentation.readthedocs.io][8]).

---

## Acknowledging Bede & Good Citizenship <a name="acknowledging-bede--good-citizenship"></a>

> “This work made use of the facilities of the N8 Centre of Excellence in Computationally Intensive Research (N8 CIR) provided and funded by the N8 research partnership and EPSRC (Grant EP/T022167/1). The Centre is co‑ordinated by the Universities of Durham, Manchester and York.”([bede-documentation.readthedocs.io][2])

Using the above text in papers, posters and talks helps justify continued funding.  Also:

* Respect shared login nodes—compile, edit, submit jobs; no long CPU/GPU runs there([bede-documentation.readthedocs.io][2]).
* Clean `$TMPDIR`; parallel I/O on Lustre saturates at 10 GB s‑¹—stage large reads/writes appropriately.

---

## Troubleshooting & Support Routes <a name="troubleshooting--support-routes"></a>

1. **Bede Support Group** on N8 CIR Slack / email `support@bede.dur.ac.uk`.
2. **Bede User Group (BUG)** forum for announcements, Q\&A and community tips.
3. **SAFE** portal → *Projects → Request password reset* if you lose MFA token.
4. Weekly login‑node “at‑risk” window (Tue 08:00–10:00) is the first place to check if sessions suddenly drop([bede-documentation.readthedocs.io][2]).

---

*Document last updated 25 July 2025.  For the canonical source and future changes, always cross‑check the live documentation.*

[1]: https://n8cir.org.uk/bede/ "Bede Supercomputer"
[2]: https://bede-documentation.readthedocs.io/en/latest/usage/index.html "Using Bede — Bede Documentation"
[3]: https://bede-documentation.readthedocs.io/en/latest/hardware/index.html "Hardware — Bede Documentation"
[4]: https://wiki.x2go.org/doku.php/download%3Astart?utm_source=chatgpt.com "download:start [X2Go - everywhere@home]"
[5]: https://wiki.x2go.org/doku.php/doc%3Ainstallation%3Ax2goclient?utm_source=chatgpt.com "doc:installation:x2goclient [X2Go - everywhere@home]"
[6]: https://bede-documentation.readthedocs.io/en/latest/software/index.html "Software on Bede — Bede Documentation"
[7]: https://bede-documentation.readthedocs.io/en/latest/software/tools/apptainer.html "Apptainer — Bede Documentation"
[8]: https://bede-documentation.readthedocs.io/en/latest/software/libraries/mpi.html "MPI — Bede Documentation"
