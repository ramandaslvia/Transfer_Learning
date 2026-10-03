# Tutorial Praktikum P2 — Transfer Learning: Tangga Lantai & Manusia

RET503 Computer Vision and Deep Learning · Pertemuan 3 · Politeknik Negeri Batam

Tugas ini mengerjakan alur Praktikum P2 (slide 20–23) dengan objek **tangga lantai** dan **manusia**.
Robot melihat ke depan lalu mengklasifikasikan citra menjadi tiga kelas:

| Kelas | Arti untuk robot | Contoh isi citra |
|---|---|---|
| `tangga` | zona berbahaya, jangan maju | anak tangga naik atau turun, tepi tangga, pegangan tangga |
| `manusia` | melambat / berhenti, jaga jarak | satu atau beberapa orang berdiri, berjalan, duduk (seluruh badan atau sebagian) |
| `lantai_datar` | aman (kelas negatif) | koridor atau lantai kosong tanpa tangga dan tanpa orang |

Kelas ketiga sengaja ditambahkan. Tanpa kelas negatif, model dipaksa menjawab "tangga" atau "manusia" untuk citra apa pun, termasuk lantai kosong.

> **Jujur soal status:** dataset berasal dari kamera robot Anda, jadi belum ada di paket ini. Semua angka hasil (akurasi, waktu latih, latensi) harus berasal dari run Anda sendiri. Skrip `split.py` dan `plot_results.py` sudah saya uji dengan data sintetis. Skrip yang memakai PyTorch (`train.py`, `latency.py`, `infer.py`) lolos pemeriksaan sintaks, tetapi belum saya jalankan karena PyTorch tidak bisa dipasang di lingkungan saya. Kalau ada error saat pertama kali jalan, kirim pesan error-nya ke saya.

---

## Isi paket

```
pj1_tangga_manusia/
├── README.md          <- tutorial ini
├── DESAIN.md          <- dokumen desain awal (templat slide 18, maks. 2 halaman)
├── requirements.txt
├── common.py          <- preprocessing bersama (train & deployment)
├── capture.py         <- langkah 1: ambil citra + metadata
├── split.py           <- langkah 2: bagi train/val berdasarkan sesi
├── train.py           <- langkah 3: feature | partial | scratch
├── latency.py         <- langkah 4: ukur latensi
├── plot_results.py    <- langkah 5: tabel + grafik hasil
└── infer.py           <- contoh inferensi satu citra (deployment)
```

## 0. Persiapan

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Pelatihan 3 mode dengan ±150–300 citra cukup ringan dan selesai dalam hitungan menit di CPU laptop. GPU tidak wajib, tetapi akan mempercepat.

---

## 1. Konsep singkat

**Transfer learning** berarti memakai ulang bobot CNN yang sudah dilatih di domain sumber (ImageNet) untuk tugas baru kita (tangga vs manusia vs lantai).

Alasannya ada di slide 6. Lapisan awal CNN mempelajari tepi, warna, dan gradien. Pola ini umum, sehingga garis tegas anak tangga dan siluet orang bisa dikenali dengan fitur yang sudah ada. Lapisan akhir lebih spesifik terhadap kelas ImageNet, jadi bagian inilah yang perlu disesuaikan.

Tiga pendekatan yang dibandingkan di praktikum ini:

| Mode | Bobot awal | Yang dilatih | Kapan masuk akal |
|---|---|---|---|
| `feature` | ImageNet | `fc` saja | data sangat sedikit, domain mirip. **Titik awal proyek.** |
| `partial` | ImageNet | `layer4` + `fc` | data sedang, domain agak berbeda |
| `scratch` | acak | semua | pembanding: seberapa besar sebenarnya manfaat transfer learning? |

Dugaan awal untuk dataset kecil: `feature` dan `partial` mencapai akurasi tinggi lebih cepat dan lebih stabil daripada `scratch`. Ini baru hipotesis. Buktikan atau bantah dengan data Anda di langkah 5.

---

## 2. Langkah 1 — Ambil data (±30 menit)

Target minggu ini: **≥ 50 citra per kelas**, jadi minimal 150 citra.

```bash
python capture.py tangga terang --sesi 1 --lokasi tangga_gedungA --catatan "tangga naik"
python capture.py tangga redup  --sesi 2 --lokasi tangga_gedungA --catatan "tangga turun"
python capture.py manusia terang --sesi 1 --lokasi koridor
python capture.py lantai_datar redup --sesi 1 --lokasi koridor
```

Tekan **SPASI** untuk menyimpan frame dan **Q** untuk keluar. Hasilnya:

```
dataset_raw/
├── metadata.csv      # nama_file, kelas, tanggal, kondisi_cahaya, lokasi, sesi, catatan
├── tangga/tangga_20261001_tangga_gedungA_terang_001.png
├── manusia/...
└── lantai_datar/...
```

Kalau di pertemuan 2 Anda sudah punya file kalibrasi kamera (`.npz` berisi `K` dan `dist`), tambahkan `--calib calib.npz`. Citra akan di-*undistort* sebelum disimpan, sesuai alur slide 13. Resolusi capture harus sama dengan resolusi saat kalibrasi.

### Variasi yang wajib diambil (slide 19, disesuaikan)

| Aspek | Tangga | Manusia |
|---|---|---|
| Jarak | dekat (±0,5 m), sedang, jauh (±3 m) | dekat, sedang, jauh |
| Posisi di citra | tengah dan tepi | tengah dan tepi |
| Sudut | lurus, miring, dari bawah dan dari atas | menghadap kamera, menyamping, membelakangi |
| Cahaya | terang, redup, dekat jendela, bayangan | sama |
| Latar | tangga beton, keramik, tangga dengan pegangan | koridor, lab, pintu |
| Sulit | tangga turun (terlihat seperti lantai), tangga gelap, motion blur | orang terpotong, 2–3 orang, orang memegang tas, jaket warna mirip lantai |

Tips penting:

1. **Pindahkan robot atau kamera di antara jepretan.** Jangan menekan SPASI 50 kali dari posisi yang sama, karena itu 50 duplikat.
2. **Ganti orang.** Pakai minimal 3–4 orang berbeda dengan baju berbeda supaya model tidak hanya menghafal satu orang.
3. **Satu citra, satu kelas dominan.** Jangan ambil citra orang berdiri di tangga untuk P2 ini, karena label tunggal jadi ambigu. Kasus gabungan lebih tepat untuk object detection (YOLO, minggu 6).
4. **Minimal 2 sesi atau 2 kondisi cahaya per kelas.** Ini syarat agar `split.py` bisa memisahkan tanpa leakage.

### Keselamatan dan privasi

- **Keselamatan:** jangan menaruh robot atau memegang kamera di tepi anak tangga teratas. Ambil data tangga turun dari jarak aman dan jangan berdiri di jalur lalu lintas orang.
- **Privasi:** foto manusia berarti data pribadi. Minta izin lisan (lebih baik tertulis) dari setiap orang yang difoto. Jika dataset diunggah ke GitHub publik, pertimbangkan memburamkan wajah atau pakai repo privat.

---

## 3. Langkah 2 — Bagi train / val (±10 menit)

```bash
python split.py
```

Skrip ini membaca `metadata.csv` dan memecah data per **sesi**, bukan per citra acak. Satu grup `(tanggal, kondisi_cahaya, sesi)` hanya masuk ke satu sisi, train atau val.

Kenapa tidak acak per citra? Itulah **data leakage** (slide 22). Frame beruntun dari posisi hampir sama akan masuk ke train dan val sekaligus. Akurasi validasi jadi terlihat bagus, tetapi model gagal di lapangan. Dengan split per sesi, val benar-benar berisi situasi yang belum pernah dilihat model.

Keluarannya `dataset/train/<kelas>`, `dataset/val/<kelas>`, dan `dataset/split_report.csv`. Jika ada peringatan "val hanya N citra", tambah data atau tambah sesi.

---

## 4. Langkah 3 — Latih tiga pendekatan (±50 menit)

```bash
python train.py feature
python train.py partial
python train.py scratch
```

Opsi umum: `--epochs 10 --batch 16 --data dataset --workers 2`. Jalankan ketiganya dengan seed dan epoch yang sama supaya perbandingannya adil. Run pertama akan mengunduh bobot ResNet-18 ImageNet (±45 MB), jadi perlu internet.

### Cara kerja `train.py` (potongan kode yang penting)

**a. Memuat model dan mengganti head**

```python
m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)   # scratch: weights=None
m.fc = nn.Linear(m.fc.in_features, num_classes)                      # head baru: 3 kelas
```

ResNet-18 asli punya 1000 keluaran (kelas ImageNet). `fc` diganti dengan layer baru berkeluaran 3. Layer baru dibuat dengan `requires_grad=True`, jadi otomatis bisa dilatih.

**b. Layer freezing**

```python
for name, p in m.named_parameters():
    p.requires_grad = name.startswith("fc.")                       # mode feature
    # mode partial: name.startswith(("layer4.", "fc."))
```

`requires_grad=False` artinya bobot itu tidak dihitung gradiennya, sehingga tidak berubah.

**c. BatchNorm lapisan beku harus tetap `eval()`** (slide 11)

```python
m.train()
for name, child in m.named_children():
    if name != "fc":
        child.eval()          # statistik BatchNorm ImageNet tidak ikut berubah
```

Jika tidak, `m.train()` membuat BatchNorm di lapisan "beku" tetap memperbarui rata-rata dan variansnya dari batch kecil kita, dan itu merusak fitur pretrained.

**d. Optimizer hanya menerima parameter yang dilatih**

```python
# feature
Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
# partial: discriminative learning rate
Adam([{"params": m.layer4.parameters(), "lr": 1e-4},
      {"params": m.fc.parameters(),     "lr": 1e-3}])
```

Pada `partial`, `layer4` memakai lr 10× lebih kecil karena bobotnya sudah bagus dan hanya perlu digeser sedikit. `fc` yang baru butuh lr lebih besar untuk belajar cepat.

**e. Augmentasi dan scheduler** (slide 21)

- Train memakai `RandomResizedCrop`, `HorizontalFlip`, dan `ColorJitter`. Ini membantu karena datanya sedikit.
- Val memakai `Resize(224,224)` + normalisasi saja.
- 10 epoch dengan `CosineAnnealingLR`.

**f. Preprocessing sama dengan model sumber** (slide 13)

Input 224×224, RGB, dan normalisasi dengan mean `0,485·0,456·0,406` dan std `0,229·0,224·0,225`. Semuanya ada di `common.py`. `ImageFolder` membaca gambar lewat PIL, jadi sudah RGB. Hati-hati kalau Anda memakai OpenCV langsung, karena OpenCV membaca BGR.

### Yang dicatat otomatis

Di `results/<mode>/`: `history.csv` (loss dan akurasi per epoch), `summary.json` (akurasi val terbaik, waktu latih, epoch pertama dengan akurasi ≥ 90%), `confusion.csv`, dan `best.pt`.

---

## 5. Langkah 4 — Ukur latensi (±15 menit)

```bash
python latency.py --classes 3 --threads 4
```

Jalankan **di perangkat target** (Raspberry Pi atau Jetson di robot). Skrip membandingkan ResNet-18 dengan MobileNetV3-Small, mengukur rata-rata, median, dan p95, lalu mengecek apakah p95 ≤ 35 ms (jatah inferensi dari anggaran 15 FPS ≈ 67 ms pada slide 16).

Kalau Anda hanya menjalankannya di laptop, tulis itu di README tugas agar tidak dianggap angka Pi. Angka ini juga hanya mencakup *forward* model. Total latensi masih ditambah akuisisi, undistort, preprocess, postprocess, dan ROS2.

---

## 6. Langkah 5 — Analisis (±15 menit)

```bash
python plot_results.py
```

Hasilnya `results/tabel_hasil.md` (tabel 3 mode) dan `results/akurasi_per_epoch.png` (grafik akurasi per epoch).

Jawab di README tugas Anda, dengan angka dari run sendiri:

1. Mode mana yang akurasi val-nya tertinggi? Mode mana yang pertama menembus 90%?
2. Berapa selisih waktu latih `feature` / `partial` dibanding `scratch`?
3. Lihat `confusion.csv`. **Kesalahan apa yang paling sering?** Untuk robot, `tangga` yang dikira `lantai_datar` jauh lebih berbahaya daripada sebaliknya, jadi laporkan recall kelas `tangga` secara khusus.
4. Apakah ada tanda *overfitting* (akurasi train tinggi, val jauh lebih rendah)? Apakah ada tanda leakage?
5. Model mana yang Anda pilih untuk robot, ResNet-18 atau MobileNetV3-Small, dan kenapa? Pertimbangkan akurasi, latensi, dan daya.

Contoh membaca confusion matrix (baris = label benar, kolom = prediksi):

```
benar\prediksi   lantai_datar  manusia  tangga
lantai_datar          18          0        2     <- 2 lantai dikira tangga (aman, hanya robot jadi hati-hati)
manusia                1         20        0
tangga                 4          0       16     <- 4 tangga dikira lantai (BERBAHAYA)
```

Angka di atas hanya ilustrasi format, bukan hasil sebenarnya.

---

## 7. Mencoba model hasil latihan

```bash
python infer.py results/feature/best.pt foto_uji.png
```

Urutannya: baca → (undistort) → BGR→RGB → resize 224 → normalisasi. Urutan ini sama dengan saat pelatihan, dan itu wajib. Pada deployment di robot, ganti `cv2.imread` dengan frame kamera (atau callback ROS2) dan pertahankan langkah preprocess-nya.

---

## 8. Pengumpulan (LMS, tautan GitHub)

Sesuai slide 23, repo harus berisi:

- [ ] Dokumen desain awal per kelompok (`DESAIN.md`; ubah ke PDF jika diminta, maks. 2 halaman)
- [ ] `dataset_raw/` + `metadata.csv`, ≥ 50 citra per kelas
- [ ] Tabel hasil 3 mode + grafik akurasi per epoch (`results/`)
- [ ] Latensi model yang dipilih (`results/latency_results.csv`)
- [ ] Analisis singkat di README

`.gitignore` sudah mengecualikan `dataset/` (bisa dibuat ulang dengan `split.py`) dan `best.pt` (besar).

## 9. Masalah yang sering muncul

| Gejala | Penyebab / solusi |
|---|---|
| `Kamera tidak bisa dibuka` | coba `--source 1`, tutup aplikasi lain yang memakai kamera, atau pakai file video |
| `split.py` bilang kelas hanya punya 1 sesi | ambil data lagi dengan `--sesi` atau kondisi cahaya berbeda |
| Akurasi val 100% sejak epoch 1–2 | curigai leakage atau val terlalu sedikit; cek `split_report.csv`, tambah sesi |
| `scratch` hampir sama bagusnya dengan `feature` | mungkin dataset terlalu mudah (latar sangat berbeda antar kelas); tambah variasi yang sulit |
| Akurasi val naik-turun tajam | val terlalu kecil; tambah data val atau tambah sesi |
| `CUDA out of memory` | `--batch 8` |
| Hasil bagus di laptop tapi buruk di robot | cek preprocessing deployment, urutan kanal BGR/RGB, kondisi cahaya baru, dan resolusi |
| Error saat unduh bobot ImageNet | perlu internet pada run pertama; bobot lalu tersimpan di cache torch |

## 10. Pengembangan setelah P2

- Tangga dan manusia **muncul bersamaan** → ganti ke object detection (YOLO) di minggu 6, dengan data dianotasi di pertemuan 4.
- Bedakan **tangga naik** dan **tangga turun**. Tangga turun adalah kasus paling berbahaya bagi robot beroda.
- Jangan jadikan klasifikasi citra ini satu-satunya pengaman. Sensor tepi (*cliff sensor*) atau ToF/ultrasonik tetap diperlukan sebagai cadangan.


# Hasil Praktikum P2 — Transfer Learning: Tangga Lantai & Manusia

**By Selvia Ramanda (4222411057)**

## Ringkasan

Proyek ini membangun klasifikator citra untuk persepsi robot beroda indoor. Dari kamera depan, model memutuskan apa yang ada di depan robot (alur Praktikum P2, slide 20–23):

| Kelas | Arti untuk robot | Contoh isi citra |
|---|---|---|
| `tangga` | zona berbahaya, jangan maju | anak tangga naik atau turun, tepi tangga, pegangan tangga |
| `manusia` | melambat / berhenti, jaga jarak | satu atau beberapa orang berdiri, berjalan, duduk (seluruh badan atau sebagian) |
| `lantai_datar` | aman (kelas negatif) | koridor atau lantai kosong tanpa tangga dan tanpa orang |

Kelas ketiga sengaja ditambahkan. Tanpa kelas negatif, model dipaksa menjawab "tangga" atau "manusia" untuk citra apa pun, termasuk lantai kosong. Satu citra dianggap satu kelas dominan. Kasus tangga dan manusia dalam satu citra tidak dicakup dan dilanjutkan dengan object detection (YOLO) di minggu 6.

Model dasarnya **ResNet-18 pretrained ImageNet**. Tiga pendekatan dibandingkan: *feature extraction*, *fine-tuning parsial*, dan *pelatihan dari nol*.

**Hasil singkat** (pada 77 citra validasi, mode split `blok`, lihat keterbatasan):

- Fine-tuning parsial: akurasi val 98,7%. Feature extraction: 97,4%. Selisih keduanya satu citra, jadi tidak disimpulkan salah satunya lebih akurat.
- Pelatihan dari nol lebih lambat dan tidak stabil: 87,0% pada run yang tersimpan (96,1% pada run sebelumnya, sehingga hasilnya tidak konsisten).
- Pada feature dan parsial, tidak ada citra `tangga` yang diprediksi sebagai `lantai_datar`.
- Latensi inferensi di laptop CPU: ResNet-18 ≈ 13 ms, MobileNetV3-Small ≈ 6 ms. Keduanya di bawah jatah 35 ms. Ini bukan angka Raspberry Pi/Jetson.

## Isi repo

```
.
├── README.md          laporan ini
├── DESAIN.md          dokumen desain awal (templat slide 18, maks. 2 halaman)
├── requirements.txt
├── common.py          preprocessing bersama (training dan deployment)
├── capture.py         langkah 1: ambil citra + metadata
├── split.py           langkah 2: bagi train/val (mode sesi atau blok)
├── train.py           langkah 3: feature | partial | scratch
├── latency.py         langkah 4: ukur latensi
├── plot_results.py    langkah 5: tabel + grafik hasil
├── infer.py           inferensi satu citra
├── calib_baru.npz     kalibrasi kamera (K, dist)
├── dataset_raw/       data mentah + metadata.csv
└── results/           hasil pelatihan, tabel, grafik, latensi
```

## Lingkungan

| Komponen | Keterangan |
|---|---|
| Sistem | Windows, VS Code, CPU saja (tanpa GPU) |
| Python | 3.13 |
| PyTorch / torchvision | 2.14.1 (CPU) / 0.29.1 |
| OpenCV | 5.0.0 |
| Perangkat latensi | laptop, 14 thread CPU. **TODO:** tulis tipe CPU. Bukan Raspberry Pi/Jetson |

## Cara menjalankan ulang

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python split.py
python train.py feature --workers 0
python train.py partial --workers 0
python train.py scratch --workers 0
python plot_results.py
python latency.py --classes 3
```

Mengambil data baru (tekan **SPASI** untuk menyimpan frame, **Q** untuk keluar):

```powershell
python capture.py tangga terang --sesi 1 --lokasi tangga_gedungA --calib calib_baru.npz
```

Menguji satu citra:

```powershell
python infer.py results/feature/best.pt contoh.png --calib calib_baru.npz
```

Run pertama `train.py` mengunduh bobot ResNet-18 ImageNet (±45 MB), jadi perlu internet.

---

## Data

Seluruh data diambil pada 3 Oktober 2026 dengan `capture.py`, resolusi 640×480.

| Kelas | Total | Sesi 1 (terang) | Sesi 2 (redup) | Lokasi |
|---|---|---|---|---|
| `tangga` | 90 | 47 | 43 | tangga_gedungA |
| `manusia` | 123 | 48 | 75 | koridor |
| `lantai_datar` | 68 | 36 | 32 | koridor |

Total 281 citra. Setiap kelas melewati target ≥ 50 citra.

- **Perangkat pengambilan:** **TODO** (webcam laptop / kamera robot / video HP).
- **Jumlah orang pada kelas `manusia`:** **TODO**.
- **Metadata:** `dataset_raw/metadata.csv` berisi `nama_file, kelas, tanggal, kondisi_cahaya, lokasi, sesi, catatan`.
- **Kalibrasi kamera:** `calib_baru.npz` (K dan dist, resolusi 640×480, RMS 0,74 px). Semua citra di dataset diambil dengan `--calib calib_baru.npz`, sehingga sudah di-*undistort* sebelum disimpan.
- **Privasi:** **TODO** (tulis bahwa orang yang difoto sudah memberi izin, dan apakah wajah diburamkan atau repo bersifat privat).

### Pembagian train / val

`split.py` memisahkan data agar frame beruntun yang mirip tidak masuk ke train dan val sekaligus (*data leakage*, slide 22). Ada dua mode, dan `auto` memilih sendiri:

| Mode | Cara kerja | Dipakai kalau |
|---|---|---|
| `sesi` | satu grup `(tanggal, kondisi_cahaya, sesi)` utuh masuk ke satu sisi | tiap kelas punya ≥ 4 sesi |
| `blok` | tiap sesi dipotong jadi blok 10 frame berurutan; blok utuh masuk ke satu sisi, dan setiap sesi menyumbang ke train maupun val | sesi < 4 per kelas |

Dataset ini hanya punya 2 sesi per kelas, sehingga **mode `blok` terpakai**. Alasannya: dengan mode `sesi`, train hanya berisi satu kondisi cahaya dan val kondisi lainnya, bahkan berbeda antarkelas (mis. tangga=terang, manusia=redup), sehingga model bisa belajar pintasan dari cahaya, bukan dari objeknya.

| Kelas | Train | Val | Train (redup/terang) | Val (redup/terang) |
|---|---|---|---|---|
| `lantai_datar` | 41 | 27 | 22 / 19 | 10 / 17 |
| `manusia` | 97 | 26 | 59 / 38 | 16 / 10 |
| `tangga` | 66 | 24 | 29 / 37 | 14 / 10 |
| **Total** | **204** | **77** | | |

Kedua kondisi cahaya muncul di train dan val untuk setiap kelas. Jumlah citra per kelas tidak seimbang (train 41 / 97 / 66) dan tidak ada pembobotan kelas, sehingga recall dilaporkan per kelas.

---

## Metode

**Konsep.** Transfer learning memakai ulang bobot CNN dari domain sumber (ImageNet) untuk tugas baru. Lapisan awal CNN mempelajari tepi, warna, dan gradien yang bersifat umum. Lapisan akhir lebih spesifik terhadap kelas ImageNet, jadi bagian itulah yang disesuaikan.

- **Model:** ResNet-18 pretrained ImageNet (`IMAGENET1K_V1`); `fc` diganti menjadi 3 keluaran (`scratch`: bobot acak).
- **Preprocessing:** undistort → resize 224×224 → RGB → normalisasi ImageNet (mean 0,485 / 0,456 / 0,406, std 0,229 / 0,224 / 0,225). Urutan sama untuk pelatihan dan inferensi (`common.py`). `ImageFolder` membaca gambar lewat PIL (RGB); OpenCV membaca BGR, jadi `infer.py` mengonversinya.
- **Augmentasi (train saja):** `RandomResizedCrop(224, scale 0,6–1,0)`, `HorizontalFlip`, `ColorJitter`.
- **Pelatihan:** 10 epoch, batch 16, Adam, `CosineAnnealingLR`, cross-entropy, seed 42, CPU.

| Mode | Bobot awal | Yang dilatih | Parameter dilatih | Learning rate |
|---|---|---|---|---|
| `feature` | ImageNet | `fc` saja | 1.539 | 1e-3 |
| `partial` | ImageNet | `layer4` + `fc` | 8.395.267 | 1e-4 / 1e-3 |
| `scratch` | acak | semua | 11.178.051 | 1e-3 |

Potongan kode yang penting di `train.py`:

```python
# layer freezing (mode feature; partial: name.startswith(("layer4.", "fc.")))
for name, p in m.named_parameters():
    p.requires_grad = name.startswith("fc.")

# BatchNorm lapisan beku tetap eval() agar statistik ImageNet tidak berubah (slide 11)
m.train()
for name, child in m.named_children():
    if name != "fc":
        child.eval()

# partial: discriminative learning rate
Adam([{"params": m.layer4.parameters(), "lr": 1e-4},
      {"params": m.fc.parameters(),     "lr": 1e-3}])
```

---

## Hasil

### Akurasi dan waktu latih

| Mode | Val terbaik (epoch) | Val epoch 10 | Epoch pertama ≥ 90% | Waktu latih |
|---|---|---|---|---|
| `feature` | 97,4% (5) | 97,4% | 3 | 59 s |
| `partial` | 98,7% (4) | 98,7% | 1 | 66 s |
| `scratch` | 87,0% (10) | 87,0% | tidak tercapai | 110 s |

Tabel lengkap ada di `results/tabel_hasil.md`.

Akurasi per epoch
<img width="1050" height="630" alt="akurasi_per_epoch" src="https://github.com/user-attachments/assets/4a839ed2-8bb5-4206-81c2-d3a787076f39" />


Catatan: `results/` berisi run terakhir. Pada run `scratch` sebelumnya (dengan seed dan konfigurasi sama, hasil tidak disimpan) akurasi val terbaik 96,1% pada epoch 6 tetapi 89,6% pada epoch 10, dengan waktu latih 115 s. Pada run `partial` pertama, waktu latih 80 s (run terakhir 66 s); akurasinya sama. Angka run sebelumnya diambil dari log terminal.

### Confusion matrix (baris = label benar, kolom = prediksi)

| Mode | Benar \ Prediksi | lantai_datar | manusia | tangga | Recall |
|---|---|---|---|---|---|
| `feature` | lantai_datar | 25 | 1 | 1 | 92,6% |
| | manusia | 0 | 26 | 0 | 100% |
| | tangga | 0 | 0 | 24 | 100% |
| `partial` | lantai_datar | 26 | 0 | 1 | 96,3% |
| | manusia | 0 | 26 | 0 | 100% |
| | tangga | 0 | 0 | 24 | 100% |
| `scratch` | lantai_datar | 21 | 3 | 3 | 77,8% |
| | manusia | 0 | 26 | 0 | 100% |
| | tangga | 0 | 4 | 20 | 83,3% |

### Latensi inferensi

Laptop CPU (14 thread), input 1×3×224×224, 200 pengukuran setelah 30 pemanasan. Hanya mencakup *forward* model.

| Model | Param (juta) | Mean | Median | p95 | FPS (dari mean) | p95 ≤ 35 ms |
|---|---|---|---|---|---|---|
| ResNet-18 | 11,18 | 12,7 ms | 12,6 ms | 14,3 ms | ≈ 79 | ya |
| MobileNetV3-Small | 1,52 | 5,7 ms | 5,0 ms | 7,6 ms | ≈ 175 | ya |

Dua pengukuran berturut-turut berbeda sekitar ±1 ms (run lain: ResNet-18 13,8 ms, MobileNetV3-Small 5,9 ms), jadi angka ini sebaiknya dibaca sebagai ≈ 13 ms dan ≈ 6 ms. Data mentah ada di `results/latency_results.csv`.

---

## Analisis

**1. Mode mana yang tertinggi dan tercepat menembus 90%?** `partial` mencapai 98,7% dan sudah ≥ 90% pada epoch 1, `feature` 97,4% pada epoch 3, `scratch` 87,0% dan tidak menembus 90% pada run tersimpan (pada run sebelumnya baru epoch 6).

**2. Selisih waktu latih.** `feature` 59 s dan `partial` 66 s, sedangkan `scratch` 110 s, sekitar 1,7–1,9× lebih lama. Waktu itu termasuk validasi tiap epoch dan diukur di CPU laptop.

**3. Kesalahan yang paling sering.** Pada `feature` dan `partial`, seluruh kesalahan terjadi pada `lantai_datar` yang dikira `manusia` atau `tangga` (arah yang aman bagi robot: berhenti atau melambat tanpa alasan). Tidak ada `tangga` yang diprediksi sebagai `lantai_datar` pada ketiga mode, dan recall `tangga` 100% (24/24) pada `feature` dan `partial`. Pada `scratch`, recall `tangga` 83,3% (di bawah target 95% di `DESAIN.md`), dengan 4 tangga dikira `manusia`. Penyebab visual kesalahan `lantai_datar`: **TODO** (lihat citra val yang salah, mis. bayangan, pantulan lantai, kaki orang di tepi gambar).

**4. Overfitting dan leakage.** Tidak ada tanda overfitting: pada `feature` dan `partial` akurasi val setara atau lebih tinggi daripada akurasi train (akurasi train dihitung dengan augmentasi menyala). Pada `scratch` akurasi train dan val sama-sama rendah dan naik-turun tajam (val sempat 32,5% pada epoch 3 dan 49,4% pada epoch 6 di run terakhir). Dugaan penyebabnya statistik BatchNorm yang tidak stabil pada data sedikit dan batch kecil; ini belum diverifikasi. Leakage antar-frame dikurangi dengan split berbasis blok, tetapi tidak sepenuhnya hilang karena val berasal dari sesi yang sama dengan train (lihat keterbatasan).

**5. Model terpilih untuk robot.** ResNet-18 dengan fine-tuning parsial: akurasi val tertinggi, mencapai 90% paling cepat, dan recall `tangga` 100% pada val. Selisihnya dengan `feature` satu citra dari 77, jadi `feature` (hanya 1.539 parameter dilatih) juga sah dipilih bila biaya pelatihan jadi pertimbangan. MobileNetV3-Small baru diukur latensinya dan **belum dilatih**, sehingga belum ada angka akurasinya. Ia kandidat deployment di Raspberry Pi karena sekitar 2,2× lebih cepat dan 7× lebih kecil, tetapi akurasi dan latensinya di perangkat target masih harus diuji.

## Keterbatasan

- **Val kecil:** 77 citra (24 tangga). Selisih beberapa poin persen tidak bermakna, dan "0 tangga terlewat" dari 24 citra belum membuktikan recall setinggi itu secara umum.
- **Val optimis:** mode `blok` membuat val berasal dari sesi yang sama dengan train. Dataset juga hanya 2 sesi per kelas, 1 hari, 1 kamera, dan hanya lokasi tertentu.
- **Epoch terbaik dipilih dari val itu sendiri,** sehingga angka "val terbaik" sedikit optimis. Angka epoch 10 dicantumkan sebagai pembanding.
- **`scratch` tidak stabil antar run** dan hanya dijalankan 2 kali. Perbandingan yang kuat butuh beberapa seed.
- **Kelas tidak seimbang** (train 41 / 97 / 66) tanpa pembobotan kelas.
- **Latensi diukur di laptop,** bukan di perangkat robot, dan hanya mencakup *forward* model. Anggaran penuh juga mencakup akuisisi, undistort, preprocess, postprocess, dan ROS2.
- **Bukan pengaman tunggal.** Sensor tepi (*cliff sensor*) atau ToF/ultrasonik tetap diperlukan sebagai cadangan.

## Pengembangan

- Ambil sesi baru (hari, lokasi, dan orang berbeda) agar `split.py` bisa memakai mode `sesi` yang lebih ketat.
- Latih MobileNetV3-Small dan ukur akurasi serta latensinya di Raspberry Pi/Jetson.
- Bedakan tangga naik dan tangga turun. Tangga turun adalah kasus paling berbahaya bagi robot beroda.
- Tangga dan manusia muncul bersamaan → ganti ke object detection (YOLO) di minggu 6.

## Pengumpulan (LMS, tautan GitHub)

- [x] Dokumen desain awal (`DESAIN.md`; ubah ke PDF jika diminta, maks. 2 halaman)
- [x] `dataset_raw/` + `metadata.csv`, ≥ 50 citra per kelas
- [x] Tabel hasil 3 mode + grafik akurasi per epoch (`results/`)
- [x] Latensi model yang dipilih (`results/latency_results.csv`)
- [x] Analisis singkat (README ini)
- [ ] Semua **TODO** di README terisi (perangkat pengambilan, jumlah orang, tipe CPU, penyebab kesalahan, privasi)

`.gitignore` mengecualikan `dataset/` (bisa dibuat ulang dengan `split.py`) dan `best.pt` (besar).
