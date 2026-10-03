# Dokumen Desain Awal — Pipeline Persepsi Pj1

**Mata kuliah:** RET503 · CDIO Stage #2 (Design) · **Kelompok:** ______ · **Tanggal:** ______

> Bagian bertanda *(asumsi)* saya isi sebagai titik awal. Sesuaikan dengan robot dan perangkat kelompok Anda.

## 1. Misi proyek
Robot beroda indoor bergerak di koridor dan area gedung bertingkat. Persepsi kamera depan harus menjawab satu pertanyaan: *apa yang ada di depan robot?* Jawabannya memicu aksi navigasi: **tangga** → berhenti dan jangan maju, **manusia** → melambat atau berhenti dan jaga jarak, **lantai datar** → lanjut. Persepsi ini pelengkap, bukan pengaman tunggal. Sensor tepi/jarak tetap menjadi cadangan.

## 2. Kelas objek

| Kelas | Deskripsi | Contoh foto |
|---|---|---|
| `tangga` | anak tangga naik/turun, tepi tangga, pegangan | *(tempel dari dataset_raw)* |
| `manusia` | satu atau beberapa orang, seluruh badan atau sebagian | *(tempel)* |
| `lantai_datar` | koridor/lantai tanpa tangga dan manusia (kelas negatif) | *(tempel)* |

Batasan P2: satu citra = satu kelas dominan. Kasus orang di atas tangga ditunda ke detection (YOLO, minggu 6).

## 3. Kamera & dudukan *(asumsi)*
USB webcam 640×480 @ 30 FPS · dipasang di depan robot, tinggi ±25–30 cm, menunduk ±20–30° agar tepi tangga dan lantai di depan terlihat · jarak kerja 0,5–3 m · lensa dikoreksi (undistort) memakai kalibrasi pertemuan 2.

## 4. Unit komputasi *(asumsi)*
Raspberry Pi 4B 4 GB, CPU saja, daya normal 5 V/3 A, inferensi PyTorch (konversi ONNX direncanakan minggu 10–11). Alternatif jika kurang cepat: Jetson.

## 5. Target kinerja

| Metrik | Target | Catatan |
|---|---|---|
| Akurasi validasi (split per sesi) | ≥ 90% | |
| Recall kelas `tangga` | ≥ 95% | kesalahan paling berbahaya |
| FPS onboard | ≥ 15 (≤ 67 ms/frame) | anggaran: inferensi ≤ 35 ms |
| Latensi ROS2 | ≤ 5 ms | |

mAP tidak dipakai di tahap ini karena tugasnya klasifikasi. mAP muncul saat beralih ke detection.

## 6. Kandidat model

| Model | Alasan |
|---|---|
| ResNet-18 (≈11,7 juta param, ≈1,8 GFLOPs) | baseline praktikum, mudah dibandingkan dengan literatur |
| MobileNetV3-Small (≈2,5 juta param, ≈0,06 GFLOPs) | kandidat deployment di Pi tanpa GPU |

Pilihan akhir ditentukan dari akurasi + `latency.py` di perangkat target.

## 7. Strategi transfer learning
Mulai dari **feature extraction** (backbone beku, hanya head dilatih) karena data masih sedikit (≥ 50 citra/kelas) dan fitur tepi/garis/siluet bersifat umum. Bandingkan dengan **fine-tuning parsial** (layer4 + head, lr 1e-4/1e-3) dan **scratch** sebagai pembanding. Jika akurasi < 90% atau recall `tangga` rendah → buka lebih banyak lapisan dan perbanyak augmentasi. Domain target (kamera rendah, indoor) cukup berbeda dari foto ImageNet, jadi *negative transfer* dipantau.

## 8. Rencana data

| Item | Rencana |
|---|---|
| Jumlah | minggu ini ≥ 50 citra/kelas (150 total); target akhir ≥ 150/kelas |
| Sesi | ≥ 2 sesi/kondisi cahaya per kelas, split per sesi |
| Variasi | jarak dekat–jauh · tengah/tepi · sudut · terang/redup/jendela/bayangan · beberapa lokasi · ≥ 3 orang berbeda · tangga naik & turun |
| Struktur | `dataset_raw/<kelas>/…png` + `metadata.csv` |

## 9. Risiko dan mitigasi

| # | Risiko | Mitigasi |
|---|---|---|
| 1 | Tangga turun terlihat seperti lantai datar dari kamera rendah (*false negative* berbahaya) | sudut kamera menunduk, banyak sampel tangga turun, pantau recall `tangga`, sensor tepi sebagai cadangan |
| 2 | Data leakage dari frame beruntun | split per sesi (`split.py`), pindah posisi tiap jepretan |
| 3 | Tangga dan manusia muncul bersamaan | aturan kelas dominan di P2, evaluasi kasus sulit terpisah, lanjut ke YOLO minggu 6 |
| 4 | Model bias ke orang/baju tertentu | ≥ 3 orang, variasi pakaian, pose, dan kerumunan kecil |
| 5 | Latensi melebihi anggaran di Pi | MobileNetV3-Small, ukur dengan `latency.py` di Pi, ONNX/TensorRT nanti |
| 6 | Privasi wajah dalam dataset | izin dari orang yang difoto, wajah diburamkan atau repo privat |
| 7 | Preprocessing training ≠ deployment | satu `common.py`, urutan undistort → BGR→RGB → resize → normalisasi |
