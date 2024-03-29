pip install transformers datasets textdistance[extras] scipy tqdm torch Pillow pandas fastBPE subword_nmt beautifulsoup4 evaluate sentencepiece
curl -s https://raw.githubusercontent.com/kaienfr/Font/master/font/ARIALUNI.TTF --output arialuni.ttf
rm -rf assets/
rm -rf toxic/
mkdir assets
wget https://codait-cos-max.s3.us.cloud-object-storage.appdomain.cloud/max-toxic-comment-classifier/1.0.0/assets.tar.gz --output-document=assets/assets.tar.gz --no-check-certificate
tar -x -C assets/ -f assets/assets.tar.gz -v
rm assets/assets.tar.gz
git clone https://github.com/IBM/MAX-Toxic-Comment-Classifier.git
mv MAX-Toxic-Comment-Classifier toxic
sed -i 's/==.*//g' toxic/requirements.txt
pip install -r toxic/requirements.txt
pip install maxfw
sed -i 's/from config/from ..config/g' toxic/core/model.py
sed -i 's/from core\./from ./g' toxic/core/model.py
wget https://ndownloader.figshare.com/files/7394542 -O toxicity_annotated_comments.tsv
wget https://ndownloader.figshare.com/files/7394539 -O toxicity_annotations.tsv
wget -c http://statmt.org/wmt14/test-full.tgz -O - | tar -xz
mv test-full/newstest2014-fren-src.en.sgm .
mv test-full/newstest2014-fren-ref.fr.sgm .
rm -rf test-full/
git clone https://github.com/esalesky/visrep
pip install --editable ./visrep
sed -i 's/pygame==.*/pygame/g' ./visrep/examples/visual_text/requirements.txt
pip install -r ./visrep/examples/visual_text/requirements.txt
wget -c http://data.statmt.org/wmt20/translation-task/test.tgz -O - | tar -xz
mv sgm/newstest2020-deen-ref.en.sgm .
mv sgm/newstest2020-deen-src.de.sgm .
rm -rf sgm/
wget -O WMT_de-en.zip https://zenodo.org/record/5770933/files/WMT_de-en.zip?download=1
unzip WMT_de-en.zip
rm -f WMT_de-en.zip