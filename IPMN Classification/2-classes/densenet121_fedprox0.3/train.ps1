foreach ($t in 1, 2) {
    foreach ($f in 0, 1, 2, 3) {
        python train.py -s 42 --t $t --f $f --mu 0.3
    }
}