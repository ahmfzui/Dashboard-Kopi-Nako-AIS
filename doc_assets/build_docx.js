const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType,
  HeadingLevel, LevelFormat,
} = require("docx");

const ASSETS = __dirname;

// ---------- helpers ----------
function run(text, opts = {}) {
  return new TextRun({ text, font: "Times New Roman", size: 24, ...opts });
}

function body(children, opts = {}) {
  return new Paragraph({
    children,
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 360, lineRule: "auto", after: 200 },
    ...opts,
  });
}

function p(text) {
  return body([run(text)]);
}

function heading(text, level) {
  return new Paragraph({
    heading: level,
    children: [new TextRun({ text, font: "Times New Roman", size: 24, bold: true })],
    spacing: { before: 300, after: 200, line: 360, lineRule: "auto" },
    outlineLevel: level === HeadingLevel.HEADING_3 ? 2 : 3,
  });
}

function imgSize(file, targetW) {
  // baca dimensi asli lewat file .dim.json yang sudah kita siapkan
  const meta = JSON.parse(fs.readFileSync(path.join(ASSETS, "dims.json"), "utf8"));
  const [w, h] = meta[file];
  return { width: targetW, height: Math.round((targetW * h) / w) };
}

function figure(file, targetW, caption) {
  const data = fs.readFileSync(path.join(ASSETS, file));
  const { width, height } = imgSize(file, targetW);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: 80 },
      children: [
        new ImageRun({
          type: "png",
          data,
          transformation: { width, height },
          altText: { title: caption, description: caption, name: file },
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [new TextRun({ text: caption, font: "Times New Roman", size: 22 })],
    }),
  ];
}

// ---------- konten ----------
const children = [];

children.push(heading("IV.6.3 Dashboard Interaktif Berbasis Streamlit", HeadingLevel.HEADING_2));

children.push(p(
  "Seluruh keluaran pada tahap Deployment diintegrasikan ke dalam dashboard interaktif berbasis Streamlit yang dibangun menggunakan bahasa Python, dengan pustaka utama Streamlit untuk antarmuka pengguna, Plotly untuk visualisasi data, serta statsmodels dan python-dateutil untuk komponen analitik dinamis yang dijelaskan lebih lanjut pada subbab ini. Dashboard dirancang untuk memudahkan eksplorasi hasil analisis secara dinamis, mulai dari skor Aspect Impact Score (AIS), Diagnostic Heatmap, distribusi sentimen, pola topik BERTopic, hingga simulasi prediksi ulasan baru menggunakan model ABSA. Dengan adanya dashboard, hasil penelitian tidak hanya disajikan dalam bentuk tabel dan visualisasi statis, tetapi juga dapat ditelusuri secara interaktif oleh pengguna."
));

children.push(p(
  "Secara arsitektur pipeline, dashboard tidak menjalankan seluruh proses ABSA secara real-time pada setiap tab yang tersedia. Ulasan pelanggan yang telah melalui tahap Aspect Category Detection (ACD) dan Aspect Sentiment Classification (ASC) pada Tahap 1 dan Tahap 2 penelitian disimpan sebagai satu dataset beranotasi (dataset_mlr_ais-wow.csv) yang berisi label sentimen per aspek untuk setiap ulasan, yaitu kolom sent_product, sent_price, sent_place, sent_promotion, sent_people, sent_process, dan sent_physical_evidence, dengan nilai -1 (negatif), 0 (tidak dibahas), atau 1 (positif). Dataset inilah yang menjadi sumber data utama Tab 1 (Ringkasan AIS dan Diagnostic Heatmap). Dengan kata lain, dashboard tidak menjalankan ulang model ABSA terhadap seluruh dataset setiap kali dibuka, melainkan langsung mengagregasi label yang sudah tersedia untuk menghitung Aspect Focus Score (AFS), Net Sentiment Score (NSS), dan bobot W melalui regresi Multiple Linear Regression (MLR) yang dilatih ulang secara dinamis, sebagaimana dijelaskan pada bagian IV.6.3.1."
));

children.push(p(
  "Tab 2 (Distribusi Sentimen) dan Tab 3 (Pola Topik BERTopic) bersumber dari berkas hasil precompute yang terpisah, yaitu sentimen_per_aspek_cabang.csv, ringkasan_model_bertopic_full.csv, hasil_ctfidf_keywords_bertopic_full.csv, dan koordinat_intertopic_distance.csv. Keempat berkas tersebut dihasilkan sekali di luar dashboard pada tahap analisis sebelumnya, kemudian ditampilkan secara statis, sehingga tidak berubah ketika pengguna berinteraksi dengan filter pada Tab 1. Satu-satunya bagian yang benar-benar menjalankan inferensi model ABSA (ACD dan ASC) secara langsung (real-time) adalah Tab 4 (Prediksi Ulasan Baru), yaitu ketika pengguna memasukkan satu teks ulasan baru yang belum pernah dilihat sistem sebelumnya. Ringkasan alur data pada dashboard ditunjukkan melalui empat tab utama: (1) Overview AIS & Heatmap, (2) Distribusi Sentimen, (3) Pola Topik (BERTopic), dan (4) Prediksi Ulasan (ABSA)."
));

children.push(p(
  "Pada bagian atas dashboard, di luar keempat tab tersebut, disediakan filter rentang tanggal serta kartu ringkasan metrik berupa total cabang, total ulasan, dan jumlah cabang pada kategori Aman, Waspada, dan Kritis, yang menyesuaikan secara otomatis terhadap rentang tanggal yang dipilih pengguna. Bagian sidebar menampilkan logo Kopi Nako, panduan status AIS, serta fitur Mode Simulasi untuk pengujian penambahan cabang baru yang dijelaskan pada bagian IV.6.3.2."
));

// ===== IV.6.3.1 =====
children.push(heading("IV.6.3.1 Ringkasan AIS, Diagnostic Heatmap, dan Filter Rentang Tanggal", HeadingLevel.HEADING_3));

children.push(p(
  "Bagian atas dashboard menampilkan filter rentang tanggal berupa dua kotak pilihan, yaitu Dari dan Sampai, serta lima kartu ringkasan metrik: Total Cabang, Total Ulasan, Kritis, Waspada, dan Aman. Tampilan filter rentang tanggal beserta kartu ringkasan metrik disajikan pada Gambar IV-22."
));
children.push(...figure("fig_01_header_filter_kartu.png", 470, "Gambar IV 22 Tampilan Filter Rentang Tanggal dan Kartu Ringkasan Metrik pada Dashboard"));

children.push(p(
  "Filter rentang tanggal menggunakan satuan waktu relatif (misalnya “3 Bulan Lalu”, “1 Tahun Lalu”), mengikuti format asli data hasil scraping Google Maps yang berupa teks relatif waktu dan bukan tanggal absolut. Teks tersebut diestimasi menjadi tanggal absolut relatif terhadap tanggal saat dashboard diakses menggunakan pustaka python-dateutil. Setiap kali rentang tanggal diubah, dashboard memfilter ulasan sesuai rentang tersebut, kemudian melatih ulang model MLR (regresi Ordinary Least Squares menggunakan pustaka statsmodels) beserta pengujian nilai p-value pada taraf signifikansi 5%, sehingga bobot W, skor AFS, NSS, dan AIS dihitung kembali khusus untuk data pada rentang tersebut, tidak lagi menggunakan bobot tetap (hardcoded) seperti pada implementasi awal sistem. Hasil perhitungan ulang ini langsung tercermin pada kartu ringkasan metrik, Diagnostic Heatmap, dan seluruh visualisasi Tab 1, sehingga pengguna dapat mengamati perubahan urgensi tiap aspek dari waktu ke waktu. Apabila jumlah ulasan pada rentang yang dipilih terlalu sedikit (kurang dari 50 ulasan) untuk menghasilkan regresi yang andal, dashboard menampilkan peringatan dan kembali menggunakan hasil dari seluruh rentang data. Proses pelatihan ulang ini memanfaatkan mekanisme cache (st.cache_data) berdasarkan kombinasi rentang tanggal yang dipilih, sehingga perpindahan antar rentang yang pernah diakses sebelumnya tetap responsif tanpa perhitungan ulang dari awal."
));

children.push(p(
  "Komponen utama pada Tab 1 adalah Diagnostic Heatmap interaktif berukuran 8 cabang dan 7 aspek 7P pada kondisi dataset produksi. Warna pada heatmap disusun dalam gradasi hijau ke kuning-oranye hingga merah, konsisten dengan tiga kategori status AIS, yaitu Aman, Waspada, dan Kritis, dengan ambang batas skor AIS masing-masing 0,050 dan 0,100. Pengguna dapat melihat nilai AIS setiap kombinasi cabang dan aspek melalui tampilan angka pada sel maupun informasi tambahan ketika kursor diarahkan ke sel tertentu. Tampilan Diagnostic Heatmap beserta keterangan kategori status disajikan pada Gambar IV-23."
));
children.push(...figure("fig_02_heatmap_legend.png", 470, "Gambar IV 23 Tampilan Diagnostic Heatmap dan Keterangan Kategori Status AIS"));

children.push(p(
  "Ketiga kategori status tersebut diturunkan langsung dari komponen pembentuk skor AIS, yaitu AIS = W × AFS × (1 − NSS), dengan W adalah bobot pengaruh aspek terhadap rating hasil regresi MLR (bernilai nol apabila aspek tersebut tidak signifikan secara statistik pada taraf p < 0,05), AFS adalah proporsi ulasan yang membahas aspek tersebut terhadap total ulasan pada cabang yang bersangkutan, dan NSS adalah skor sentimen bersih (net sentiment score) pada rentang −1 hingga 1. Kategori Aman (AIS ≤ 0,050) menunjukkan aspek yang jarang dibahas pelanggan dan/atau sentimennya cenderung positif, atau pengaruhnya terhadap rating tidak signifikan secara statistik, sehingga belum memerlukan tindakan perbaikan segera. Kategori Waspada (0,051 ≤ AIS ≤ 0,100) menunjukkan aspek yang mulai cukup sering dibahas dan/atau sentimennya mulai mengarah negatif, sehingga perlu dipantau agar tidak memburuk. Kategori Kritis (AIS > 0,100) menunjukkan aspek yang berpengaruh signifikan terhadap rating, sering dibahas pelanggan, dan sentimennya didominasi ulasan negatif, sehingga menjadi prioritas utama perbaikan operasional pada cabang terkait. Penjelasan ketiga kategori ini ditampilkan langsung di bawah Diagnostic Heatmap pada dashboard agar pengguna dapat memahami makna setiap warna tanpa perlu merujuk pada dokumentasi terpisah."
));

children.push(p(
  "Berbeda dengan filter tanggal, filter Cabang dan Aspek 7P yang berada di dalam Tab 1 bersifat murni tampilan (view-only): keduanya hanya menyaring baris/kolom yang ditampilkan pada grafik batang dan tabel detail, tanpa memicu pelatihan ulang model MLR maupun mengubah Diagnostic Heatmap, karena keduanya dimaksudkan sebagai preferensi eksplorasi visual pengguna, bukan sebagai parameter analitik. Ketika filter digunakan, grafik batang dan tabel detail menyesuaikan tampilan secara otomatis: grafik batang menampilkan skor AIS sesuai pilihan pengguna, sedangkan tabel detail menampilkan cabang, aspek, skor AIS, dan statusnya. Tampilan bagian filter, grafik batang, dan tabel detail disajikan pada Gambar IV-24."
));
children.push(...figure("fig_03_detail_filter_bar.png", 470, "Gambar IV 24 Tampilan Detail Filter, Grafik Batang, dan Tabel Detail AIS"));

children.push(p(
  "Pada bagian akhir Tab 1, terdapat ringkasan prioritas kinerja cabang yang menunjukkan aspek dengan skor AIS tertinggi pada setiap cabang, diurutkan dari skor tertinggi ke terendah beserta status masing-masing. Ringkasan ini membantu pengguna mengidentifikasi dengan cepat cabang dan aspek mana yang memerlukan perhatian manajerial paling mendesak. Tampilan ringkasan prioritas kinerja cabang disajikan pada Gambar IV-25."
));
children.push(...figure("fig_04_ringkasan_prioritas.png", 470, "Gambar IV 25 Tampilan Ringkasan Prioritas Kinerja Cabang"));

// ===== IV.6.3.2 =====
children.push(heading("IV.6.3.2 Simulasi Penambahan Data dan Cabang Baru", HeadingLevel.HEADING_3));

children.push(p(
  "Untuk mengakomodasi kebutuhan operasional apabila terdapat cabang baru atau data ulasan tambahan di kemudian hari, dashboard dilengkapi fitur unggah data (upload) pada sidebar, di dalam bagian Mode Simulasi. Fitur ini memungkinkan pengguna mengunggah berkas CSV baru dengan struktur kolom yang sama dengan dataset utama, yaitu cabang, tanggal, rating, serta tujuh kolom label sentimen per aspek 7P (sent_product, sent_price, sent_place, sent_promotion, sent_people, sent_process, sent_physical_evidence), dengan kolom place_name dan ulasan bersifat opsional. Setiap berkas yang diunggah divalidasi terlebih dahulu sebelum digabungkan (append) ke data yang sedang aktif, meliputi kelengkapan kolom wajib, rentang nilai rating (1–5), serta nilai label sentimen yang harus berupa −1, 0, atau 1."
));

children.push(p(
  "Untuk keperluan pengujian, Mode Simulasi menyediakan opsi “Mulai dari dataset contoh 7 cabang”, yang mengganti dataset dasar secara sementara dari 8 cabang produksi menjadi versi contoh berisi 7 cabang, menyisihkan satu cabang dengan jumlah ulasan paling sedikit sebagai simulasi “cabang baru”, tanpa mengubah data produksi asli. Pengguna kemudian dapat mengunggah berkas ulasan cabang kedelapan tersebut untuk menguji respons dashboard terhadap penambahan cabang baru: seluruh kartu ringkasan metrik, Diagnostic Heatmap, bobot MLR, dan skor AIS pada Tab 1 dihitung ulang secara otomatis mengikuti data gabungan, tanpa memerlukan perubahan kode maupun pelatihan ulang model secara manual. Data hasil unggahan disimpan permanen pada berkas terpisah di sisi server beserta berkas log riwayatnya, sehingga tetap tersedia meskipun halaman dashboard dimuat ulang (refresh), selama Mode Simulasi masih diaktifkan. Dashboard juga menyediakan opsi “Hapus data tambahan (reset simulasi)” untuk menghapus seluruh data unggahan dari server dan mengembalikan dashboard ke kondisi dataset contoh semula, sehingga pengujian simulasi penambahan cabang dapat diulang kapan pun diperlukan. Tampilan Mode Simulasi setelah data cabang baru berhasil diunggah dan tersimpan disajikan pada Gambar IV-26."
));
children.push(...figure("fig_05b_upload_sukses.png", 180, "Gambar IV 26 Tampilan Mode Simulasi dan Fitur Unggah Data Cabang Baru pada Sidebar"));

// ===== IV.6.3.3 =====
children.push(heading("IV.6.3.3 Distribusi Sentimen per Aspek", HeadingLevel.HEADING_3));

children.push(p(
  "Tab 2 menampilkan distribusi sentimen pelanggan pada setiap aspek 7P berdasarkan data hasil precompute (sentimen_per_aspek_cabang.csv). Pengguna dapat memilih mode perbandingan dua cabang atau mode satu cabang. Secara bawaan, tab ini membandingkan Cabang Cinere dan Cabang Abdul Muis sebagai cabang objek analisis utama. Grafik batang menampilkan jumlah ulasan positif dan negatif pada setiap aspek, sehingga perbedaan tekanan sentimen antar cabang dapat diamati secara langsung. Pada bagian bawah tab, tabel detail menampilkan jumlah ulasan positif, negatif, total ulasan, dan persentase negatif. Kolom persentase negatif diberi warna untuk membantu interpretasi, yaitu merah untuk tekanan negatif tinggi (di atas 40%) dan oranye untuk aspek yang perlu diperhatikan (di atas 20%). Tampilan distribusi sentimen per aspek disajikan pada Gambar IV-27."
));
children.push(...figure("fig_06_distribusi_sentimen.png", 470, "Gambar IV 27 Tampilan Distribusi Sentimen per Aspek pada Dashboard"));

// ===== IV.6.3.4 =====
children.push(heading("IV.6.3.4 Pola Topik Pelanggan dengan BERTopic", HeadingLevel.HEADING_3));

children.push(p(
  "Tab 3 menampilkan hasil analisis pola topik pelanggan menggunakan BERTopic berdasarkan berkas hasil precompute (ringkasan_model_bertopic_full.csv, hasil_ctfidf_keywords_bertopic_full.csv, dan koordinat_intertopic_distance.csv). Pengguna dapat memilih cabang dan jenis ulasan, yaitu positif atau negatif. Setelah pilihan ditentukan, sistem menampilkan ringkasan model berupa jumlah topik, nilai coherence (Cv), jumlah dokumen berlabel, dan nilai min_cluster_size. Visualisasi utama pada tab ini berupa peta kedekatan antartopik, dengan setiap topik ditampilkan sebagai gelembung pada ruang dua dimensi. Ukuran gelembung menunjukkan jumlah dokumen dalam topik, sedangkan posisinya menunjukkan kedekatan antartopik. Di sebelah peta, sistem menampilkan kartu ringkasan topik yang berisi nomor topik, jumlah dokumen, dan kata kunci utama. Tampilan peta kedekatan antartopik disajikan pada Gambar IV-28."
));
children.push(...figure("fig_07_peta_topik.png", 470, "Gambar IV 28 Tampilan Peta Kedekatan Antartopik BERTopic"));

children.push(p(
  "Selain peta topik, tab ini juga menampilkan grafik kata kunci utama berdasarkan skor c-TF-IDF. Pengguna dapat memilih topik tertentu melalui fitur multiselect, sehingga hanya topik yang ingin dianalisis yang ditampilkan. Pada bagian bawah, tersedia tabel lengkap kata kunci dan skor c-TF-IDF dalam komponen yang dapat diperluas (expander). Tampilan kata kunci utama per topik disajikan pada Gambar IV-29."
));
children.push(...figure("fig_08_katakunci_topik.png", 470, "Gambar IV 29 Tampilan Kata Kunci Utama per Topik Berdasarkan c-TF-IDF"));

// ===== IV.6.3.5 =====
children.push(heading("IV.6.3.5 Prediksi Ulasan Baru Menggunakan ABSA", HeadingLevel.HEADING_3));

children.push(p(
  "Tab 4 menyediakan fitur simulasi prediksi ulasan baru menggunakan model ABSA yang telah dilatih pada tahap sebelumnya, dan merupakan satu-satunya bagian dashboard yang menjalankan inferensi model secara real-time. Fitur ini memungkinkan pengguna memasukkan satu teks ulasan yang belum pernah dilihat sistem sebelumnya, kemudian sistem memprosesnya melalui dua tahap, yaitu Aspect Category Detection (ACD) dan Aspect Sentiment Classification (ASC). Kedua model (IndoBERT + BiLSTM/CNN untuk ACD dan IndoBERT + BiLSTM‖CNN untuk ASC) dimuat sekali ke memori dan disimpan dalam cache (st.cache_resource) agar prediksi berikutnya tidak perlu memuat ulang model dari berkas."
));

children.push(p(
  "Pada tahap ACD, model mendeteksi aspek 7P yang muncul dalam ulasan dan menampilkan probabilitas untuk setiap aspek. Pengguna dapat mengatur nilai threshold deteksi aspek melalui slider; aspek yang probabilitasnya melampaui nilai threshold ditandai sebagai aspek terdeteksi. Selanjutnya, tahap ASC memprediksi sentimen pada setiap aspek yang terdeteksi. Hasil prediksi ditampilkan dalam bentuk kartu sentimen yang memuat label positif, netral, atau negatif beserta probabilitasnya."
));

children.push(p(
  "Sebagai ilustrasi, ulasan “Kopi susunya enak dan tidak terlalu manis, tapi proses penyajiannya agak lama saat kondisi ramai” menghasilkan deteksi aspek Product (probabilitas 100%) dan Process (probabilitas 77%) pada tahap ACD dengan threshold 0,50. Pada tahap ASC, aspek Product diprediksi Positif karena ulasan mengapresiasi rasa kopi, sedangkan aspek Process diprediksi Negatif karena terdapat keluhan terhadap lamanya proses penyajian. Tampilan fitur prediksi ulasan baru beserta hasil ilustrasi tersebut disajikan pada Gambar IV-30."
));
children.push(...figure("fig_09_absa_prediksi.png", 470, "Gambar IV 30 Tampilan Prediksi Ulasan Baru Menggunakan Model ABSA"));

// ---------- dokumen ----------
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 24 } } },
    paragraphStyles: [
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 240, after: 240 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 2 },
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 2268, right: 1701, bottom: 1701, left: 2268 },
        },
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(path.join(ASSETS, "..", "IV.6.3 Dashboard Interaktif Berbasis Streamlit.docx"), buffer);
  console.log("docx written");
});
