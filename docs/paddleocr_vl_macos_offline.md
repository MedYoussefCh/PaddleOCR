# PaddleOCR-VL offline sur macOS (Apple Silicon)

Ce guide explique comment exécuter la pipeline `PaddleOCR-VL` entièrement hors ligne sur un MacBook Pro Apple M1 Pro (macOS Sequoia 15.6.1, Python 3.12, 16 Go de RAM) en chargeant chaque modèle depuis un répertoire local.

## 1. Préparer l'environnement Python
1. Créez un environnement virtuel Python 3.12.
2. Installez PaddlePaddle CPU (recommandé sur Apple Silicon si vous n'utilisez pas de backend externe) puis PaddleOCR avec toutes les fonctionnalités :
   ```bash
   python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
   python -m pip install "paddleocr[all]"
   ```
   Ces commandes suivent le guide d'installation général de PaddleOCR.

3. Ajoutez la dépendance de rendu PDF pour l'extraction page par page :
   ```bash
   python -m pip install pypdfium2
   ```

## 2. Rassembler tous les modèles en local
Téléchargez chaque modèle sur une machine connectée, copiez-les sur le Mac et organisez-les, par exemple, sous `~/models/paddleocr_vl/` :
- `layout_detection` : modèle de détection/tri de mise en page (ex. `PP-DocLayout_plus-L`).
- `vl_rec` : VLM de reconnaissance (`PaddleOCR-VL-0.9B`).
- `doc_orientation_classify` : classification d'orientation des pages (optionnel, si `use_doc_orientation_classify=True`).
- `doc_unwarping` : redressement/dewarping (optionnel, si `use_doc_unwarping=True`).
- `chart_recognition` : extraction de graphiques (optionnel, si `use_chart_recognition=True`).

Arborescence type :
```
~/models/paddleocr_vl/
  layout_detection/        # poids + config du détecteur de layout
  vl_rec/                  # poids + config du VLM PaddleOCR-VL-0.9B
  doc_orientation_classify/# poids + config du classifieur d'orientation (si utilisé)
  doc_unwarping/           # poids + config du modèle de déwarping (si utilisé)
  chart_recognition/       # poids + config du modèle de chart (si utilisé)
```
Assurez-vous que chaque dossier contient les fichiers de configuration et de poids attendus par PaddleX/PaddleOCR.

## 3. Pipeline page par page pour un dossier `input/`
### Principe demandé
- Dossier d'entrée : `input/` contenant uniquement des PDF.
- Dossier de sortie : `output/`.
- Pour `mon_doc.pdf` : `output/mon_doc.pdf.page01.txt`, `output/mon_doc.pdf.page02.txt`, etc.

### Script prêt à l'emploi
Un script dédié est fourni dans `tools/run_paddleocr_vl_pdf_batch.py`. Il rend chaque page en image, lance `PaddleOCRVL`, récupère la sortie Markdown et l'enregistre en `.txt` page par page.

Lancer la pipeline (toutes les pages, hors ligne) :
```bash
# Variables à adapter à vos chemins
LAYOUT_DIR=~/models/paddleocr_vl/layout_detection
VLM_DIR=~/models/paddleocr_vl/vl_rec
ORI_DIR=~/models/paddleocr_vl/doc_orientation_classify   # optionnel
UNWARP_DIR=~/models/paddleocr_vl/doc_unwarping           # optionnel

python tools/run_paddleocr_vl_pdf_batch.py \
  --input_dir input \
  --output_dir output \
  --layout_detection_model_dir "$LAYOUT_DIR" \
  --vl_rec_model_name PaddleOCR-VL-0.9B \
  --vl_rec_model_dir "$VLM_DIR" \
  --doc_orientation_classify_model_dir "$ORI_DIR" \
  --doc_unwarping_model_dir "$UNWARP_DIR" \
  --use_doc_orientation_classify \
  --use_doc_unwarping
```

Le script :
- Parcourt tous les PDF de `input/`.
- Rend chaque page (réglage `--render_scale` pour affiner la résolution, par défaut `1.0` pour économiser les 16 Go de RAM).
- Appelle `PaddleOCRVL` avec vos modèles locaux (aucun téléchargement réseau).
- Écrit `output/<nom_pdf>.pageXX.txt` (XX avec au moins 2 chiffres, et davantage si le PDF comporte plus de 99 pages).

Options utiles :
- `--use_chart_recognition` si vous avez aussi le modèle de graphique.
- `--min_pixels` / `--max_pixels` pour cadrer la résolution envoyée au modèle.
- `--verbose` pour voir les logs détaillés.

## 4. Exemple d'usage Python (100 % local)
```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    layout_detection_model_dir="~/models/paddleocr_vl/layout_detection",
    vl_rec_model_name="PaddleOCR-VL-0.9B",
    vl_rec_model_dir="~/models/paddleocr_vl/vl_rec",
    doc_orientation_classify_model_dir="~/models/paddleocr_vl/doc_orientation_classify",
    doc_unwarping_model_dir="~/models/paddleocr_vl/doc_unwarping",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_chart_recognition=False,  # passez à True si vous avez le modèle de chart
)

output = pipeline.predict("/chemin/vers/mon_document.png")
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
```
Les arguments `*_model_dir` pointent vers vos répertoires locaux ; `vl_rec_model_name` définit explicitement le VLM (0.9B). Aucun téléchargement n'est tenté lorsque les répertoires sont fournis.

## 5. Exemple via la CLI
```bash
paddleocr doc_parser \
  --input /chemin/vers/mon_document.png \
  --layout_detection_model_dir ~/models/paddleocr_vl/layout_detection \
  --vl_rec_model_name PaddleOCR-VL-0.9B \
  --vl_rec_model_dir ~/models/paddleocr_vl/vl_rec \
  --doc_orientation_classify_model_dir ~/models/paddleocr_vl/doc_orientation_classify \
  --doc_unwarping_model_dir ~/models/paddleocr_vl/doc_unwarping \
  --use_doc_orientation_classify True \
  --use_doc_unwarping True \
  --use_chart_recognition False
```
Adaptez les chemins et options selon les modules que vous activez.

## 6. Conseils spécifiques à macOS (Apple Silicon)
- La version CPU de PaddlePaddle est adaptée à Apple Silicon ; l'usage reste sur CPU/MPS, ce qui limite la consommation mémoire par rapport aux builds CUDA/Linux.
- Le VLM 0.9B est plus léger que les variantes 3B ; avec 16 Go de RAM, privilégiez des images de résolution modérée et exécutez le traitement page par page.
- Conservez les modèles sur un SSD local pour réduire la latence de chargement.
