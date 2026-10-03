| Mode | Bobot awal | Yang dilatih (jumlah param) | Akurasi val terbaik | Epoch terbaik | Epoch pertama ≥ 90% | Waktu latih (detik) |
|---|---|---|---|---|---|---|
| feature | ImageNet | fc saja (1,539) | 97.4% | 5 | 3 | 59.4 |
| partial | ImageNet | layer4 + fc (8,395,267) | 98.7% | 4 | 1 | 66.4 |
| scratch | acak | semua (11,178,051) | 87.0% | 10 | tidak tercapai | 110.3 |
