import constants

def get_run_name(db_type, net_type, cutoff_layer):
  if net_type in ["frozen", "frozenv2"]:
    return "%scut%i_%s" % (net_type, cutoff_layer, db_type)
  else:
    return db_type

def is_contrast(net_type):
  return net_type in ["contrast", "frozen_contrast"]

def get_dirs(run_name, net_type):
  ckpt_dir = "sessions/%s/ckpts_%s/" % (net_type, run_name)
  summaries_dir = "sessions/%s/logs_%s/summaries" % (net_type, run_name)
  return ckpt_dir, summaries_dir

def get_settings(setting_num):
  net_type, cutoff_layer, db_type, learning_rate, batch_size, vary_ratio = \
      constants.SETTINGS[setting_num]
  run_name = get_run_name(db_type, net_type, cutoff_layer)
  ckpt_dir, summaries_dir = get_dirs(run_name, net_type)
  return run_name, ckpt_dir, summaries_dir, learning_rate, batch_size, vary_ratio
