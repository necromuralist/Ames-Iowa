from pathlib import Path

import attr
import pandas

from ames.data.constants import RawData
from ames.data.loaders import EnvironmentLoader, RawLoader


@attr.s(auto_attribs=True)
class Submission:
    """Makes a submission file for the kaggle competition

    Args:
     model: the model to make a submission
     tag: descriptive tag for the submission file name
     fit: whether the model needs to be fit or not
     data: object with the validation data
   """
    model: object
    tag: str
    fit: bool=False
    _data: RawLoader=None
    _environment: EnvironmentLoader=None
    _path: Path=None
    _predictions: pandas.DataFrame=None
    _test_path: Path=None
    _test_data: pandas.DataFrame=None

    @property
    def environment(self) -> EnvironmentLoader:
        """Loader of the environment variables"""
        if self._environment is None:
            self._environment = EnvironmentLoader()
        return self._environment

    @property
    def path(self) -> Path:
        """Path to the submission folder"""
        if self._path is None:
            self._path = Path(
                self.environment[RawData.submissions]).expanduser()
            if not self._path.is_dir():
                self._path.mkdir()
        return self._path

    @property
    def test_path(self) -> Path:
        """Path to the test set"""
        if self._test_path is None:
            self._test_data = Path(self.environment[RawData.test]).expanduser()
            assert self._test_data.is_file()
        return self._test_data

    @property
    def data(self) -> RawLoader:
        """holds the data"""
        if self._data is None:
            self._data = RawLoader()
        return self._data

    @property
    def test_data(self) -> pandas.DataFrame:
        """The test-data for submissions"""
        if self._test_data is None:
            self._test_data = pandas.read_csv(self.test_path, index_col="Id")
        return self._test_data

    @property
    def predictions(self) -> pandas.DataFrame:
        """The data-frame with the predictions"""
        if self._predictions is None:
            self._predictions = self.model.predict(self.test_data)
            self._predictions = pandas.DataFrame({RawData.primary_key: self.test_data.index,
                                                  RawData.target: self._predictions})
        return self._predictions

    def __call__(self) -> None:
        """Create the submission file"""
        if self.fit:
            self.model.fit(self.data.x_train, self.data.y_train)
        self.predictions.to_csv(self.path/f"submission_{self.tag}.csv", index=False)
        return
