Adding a dataset:
  create a txt file with a bunch of paths in indexes/train, indexes/val, indexes/test
    path/to/image1.png, path/to/image2.png
    path/to/image3.png, path/to/image4.png
    path/to/image5.png, path/to/image6.png

  Add a function that loads your datasets using SimpleDB in positive\_dbs.py or negative\_dbs.py
    def make_rgbd(folder, stats):
      return [SimpleDB("indexes/"+folder+"/DATASET_NAME.txt",
                "raw/DATASET_IMAGE_FOLDER", stats, 1)]

  Add a call to your function in load\_dbs.py with db.extend(....._dbs.make_rgbd(folder, stats))
    insert into make_db if the pairs are positive, make_n_db if they are negative

  Recalculate your dataset mean and std using calc\_image\_stats.py NAME\_OF\_NEW\_TRAINING\_SET
    If you want multiple datasets to be combined for the new training set, your new training set
    should be named SET1_SET2_SET3

  Add the new stats to constants.py
    add the name of the new training set to STATS with the new stats for the training set

  Add a new setting for the networks you want to train on the new dataset
  Train that setting

Adding a model:
  Create a script in model with the function top and bot
    top(is_training) -> l_inputs, r_inputs, l_output, r_output
  if frozen:
    top(is_training, graph_path, layer_cutoff) -> l_inputs, r_inputs, l_output, r_output
  Other parameters can be passed but loadModel must be changed to accomodate them
    bot(l_output, r_output, is_training) -> logits, result
  For contrastive loss
    bot(l_output, r_output, is_training) -> result
  The network then has to be imported to constants.py and added to NET\_TYPES
    The format is NAME_OF_NET: IMPORTED_NET
  A setting must then be added to SETTINGS so it can be trained
