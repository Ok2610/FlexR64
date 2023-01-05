# FlexR64
Ratio-64 compression and decompression scripts

## Preperation
Create files with image ids and features. The ids should start from 0 and be incremental. Therefore we suggest you keep a mapping file of image id and actual image name. The features file has to have the same ordering as the image ids.

## Compress Example
```
python compress-iota_i64.py iota out_dir hdf5_comp_features hdf5_comp_ids --feat_hdf5_name 'DATA' --p N --norm
```

For more details run python flex-compress-r64.py --help
