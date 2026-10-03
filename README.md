# Praktikum P2 — Transfer Learning: Tangga Lantai & Manusia

RET503 Computer Vision and Deep Learning · Pertemuan 3 · Politeknik Negeri Batam

**By Selvia Ramanda (4222411057)**


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
