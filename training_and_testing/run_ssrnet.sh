#max=5
#for i in $(seq 1 $max); do
#  echo "$i"
#  for j in $(seq 1 $max); do
#    echo "$j"
#    # KERAS_BACKEND=tensorflow python SSRNET_train.py --input ../data/imdb_db.npz --db imdb --netType1 $i --netType2 $j
#    KERAS_BACKEND=tensorflow python SSRNET_train.py --input ../data/wiki_db.npz --db wiki --netType1 $i --netType2 $j --batch_size 50
#    # KERAS_BACKEND=tensorflow python SSRNET_train.py --input ../data/morph2_db_align.npz --db morph --netType1 $i --netType2 $j --batch_size 50
#  done
#done

KERAS_BACKEND=tensorflow python SSRNET_train.py \
  --input ../data/wiki_db.npz \
  --db wiki \
  --netType1 4 \
  --netType2 4 \
  --batch_size 50
