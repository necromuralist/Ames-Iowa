from pathlib import Path
import os

from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
import attr
import pandas

from ames.data.constants import RawData


@attr.s(auto_attribs=True)
class EnvironmentLoader:
    """Loads the environment variable into a dictionary"""
    _environment: dict=None

    @property
    def environment(self) -> dict:
        """the environment variables"""
        if self._environment is None:
            load_dotenv(find_dotenv())
            self._environment = os.environ
        return self._environment

    def __getitem__(self, variable: str) -> str:
        """Get the value for the environment variable

        Args:
         variable: name of the environment variable to get
        """
        return self.environment.get(variable)


@attr.s(auto_attribs=True)
class RawLoader:
    """Loads the raw training data"""
    raw_data: object=RawData
    _environment: EnvironmentLoader=None
    _path: Path=None
    _data: pandas.DataFrame=None
    _X: pandas.DataFrame = None
    _y: pandas.DataFrame = None
    _x_train: pandas.DataFrame = None
    _y_train: pandas.DataFrame = None
    _x_validate: pandas.DataFrame = None
    _y_validate: pandas.DataFrame = None

    @property
    def environment(self) -> EnvironmentLoader:
        """The environment variables"""
        if self._environment is None:
            self._environment = EnvironmentLoader()
        return self._environment

    @property
    def path(self) -> Path:
        """Path to the data"""
        if self._path is None:
            self._path = Path(
                self.environment[
                    self.raw_data.environment_variable]).expanduser()
        return self._path

    @property
    def data(self) -> pandas.DataFrame:
        """The training set"""
        if self._data is None:
            self._data = pandas.read_csv(self.path)
        return self._data

    @property
    def X(self) -> pandas.DataFrame:
        """The features"""
        if self._X is None:
            self._X = self.data[[column for column in self.data.columns
                                 if column != self.raw_data.target]]
        return self._X

    @property
    def y(self) -> pandas.DataFrame:
        """The target values"""
        if self._y is None:
            self._y = self.data[self.raw_data.target]
        return self._y

    @property
    def x_train(self) -> pandas.DataFrame:
        """The training features"""
        if self._x_train is None:
            self.train_test_split()
        return self._x_train

    @property
    def x_validate(self) -> pandas.DataFrame:
        """the validation features"""
        if self._x_validate is None:
            self.train_test_split()
        return self._x_validate

    @property
    def y_train(self) -> pandas.DataFrame:
        """The training target values"""
        if self._y_train is None:
            self.train_test_split()
        return self._y_train

    @property
    def y_validate(self) -> pandas.DataFrame:
        """The validation target values"""
        if self._y_validate is None:
            self.train_test_split()
        return self._y_validate

    def train_test_split(self) -> None:
        """Splits up the data sets"""
        self._x_train, self._x_validate, self._y_train, self._y_validate = train_test_split(
            self.X, self.y, random_state=self.raw_data.random_seed,
            train_size=self.raw_data.train_size, test_size=self.raw_data.test_size)
        return
