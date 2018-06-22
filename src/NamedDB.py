# NamedDB gives databases a name so it can be printed to the terminal
class NamedDB:
  # This should be called in the __init__ of a new database by using
  # super.__init__(self, path_to_index, "DatabaseName")
  def __init__(self, path_to_index, name):
    self._path = path_to_index
    self._dbname = name

  # The name property, which can be accessed with db.name
  @property
  def name(self):
    """
    prefix = ""
    if "pitts" in self._path:
      prefix = "pitts"
    if "t27" in self._path:
      prefix = "t27"
    if "ttm" in self._path:
      prefix = "ttm"
    """
    prefix = self._path[:-4]
    prefix = prefix.split("/")[-1]

    return prefix + "_" + self._dbname
