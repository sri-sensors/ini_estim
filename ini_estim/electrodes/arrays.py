import pathlib
import re
import pandas as pd
import numpy as np
import math
import scipy.interpolate as interp
import logging
from tqdm.auto import tqdm, trange
from ini_estim import DATA_DIR
from ini_estim import stimulus
from ini_estim.utilities import sampling
import warnings

COMSOL_DIR = pathlib.Path(DATA_DIR)

# Max current values are an approximation guided by:
#
# Günter, C., Delbeke, J. & Ortiz-Catalan, M. Safety of long-term electrical 
# peripheral nerve stimulation: review of the state of the art. 
# J NeuroEngineering Rehabil 16, 13 (2019). 
# https://doi.org/10.1186/s12984-018-0474-8


class GenericElectrodeCuff:
    DATA_DIR = COMSOL_DIR / "cuff_fem" / "exports"
    FILEREG = re.compile(r"14elcuff_singleelon_1mAthrough_el([0-9]+)")
    FILEGLOB = "*.txt"
    _CX_MM = 0.0    # millimeters
    _CY_MM = 0.0    # millimeters
    DS = 0.002  # step size in mm (from inspection of data file)
    DATA_TO_MM = 1000
    CURRENT_REFERENCE_UA = 1000     # uA
    MAX_CURRENT_UA = 250
    DIAMETER_MM = 3.0
    MAX_ELECTRODES = 14

    def __init__(self):
        all_files = self.DATA_DIR.glob(self.FILEGLOB)
        self._loaded = {}
        self.files = {}
        for f in all_files:
            m = re.match(self.FILEREG, f.name)
            if m is None:
                continue
            k = int(m.groups()[0])
            self.files[k] = f.name
            self._loaded[k] = False

        self._df = {}
        self._funs = {}
        self._bounds = None
        self._active_electrodes = None

    def __str__(self):
        return "generic_cuff"

    @property
    def area(self):
        return math.pi*(self.DIAMETER_MM*0.5)**2

    @property
    def bounds(self):
        if self._bounds is None:
            self.load(self.electrodes[0])
        return self._bounds

    def _load_csv(self, filename):
        return pd.read_csv(self.DATA_DIR / filename,
                           delim_whitespace=True,
                           skiprows=9,
                           header=None,
                           names=['x', 'y', 'z', 'v'])

    def load(self, electrodes=None):
        """ Load data associated with electrodes

        Parameters
        ----------
        electrodes
            A single electrode or a list of electrodes. If unspecified,
            all electrodes will be loaded.
        """
        if electrodes is None:
            electrodes = self.electrodes

        if not isinstance(electrodes, str):
            try:
                i = iter(electrodes)
            except TypeError:
                i = None
        else:
            i = None
        if i is None:
            self._load_single(electrodes)
        else:
            e_iter = tqdm(electrodes)
            e_iter.set_description("Loading electrode data.")
            for e in e_iter:
                self._load_single(e)

    def _load_single(self, electrode):
        """
        Load the data for specified electrode

        Parameters
        ----------
        electrode
            A single electrode to load
        """

        if self._loaded[electrode]:
            return
        file = self.files[electrode]
        LOG = logging.getLogger(__name__)
        LOG.info("Loading data for file: {}".format(file))
        df = self._load_csv(file)
        z_range = df["z"].max() - df["z"].min()
        if z_range > np.finfo(float).eps*10:
            warnings.warn("Encountered data with multiple Z values!")

        xy = df[['x','y']].values*self.DATA_TO_MM - np.array([self._CX_MM, self._CY_MM])
        xymin = xy.min(axis=0)
        xymax = xy.max(axis=0)
        v = df[['v']].values

        self._df[electrode] = df
        if self._bounds is None:
            self._bounds = {
                'x': [xymin[0], xymax[0]],
                'y': [xymin[1], xymax[1]]
            }

        self._funs[electrode] = interp.NearestNDInterpolator(xy, v)
        self._loaded[electrode] = True
        return

    @property
    def electrodes(self):
        return sorted(list(self.files.keys()))

    @property
    def num_electrodes(self):
        return len(self.files)
    
    @property
    def active_electrodes(self):
        if self._active_electrodes is None:
            self._active_electrodes = self.electrodes
        return self._active_electrodes
    
    @active_electrodes.setter
    def active_electrodes(self, val):
        all_electrodes = set(self.electrodes)
        active_electrodes = set(val)
        good_electrodes = active_electrodes & all_electrodes
        bad_electrodes = active_electrodes - good_electrodes
        if len(bad_electrodes):
            warnings.warn("Invalid electrodes {} were removed.".format(
                list(bad_electrodes)))
        self._active_electrodes = sorted(list(good_electrodes))

    def get_potential(self, x, y, electrodes=None):
        """
        Retrieve the potential for a specified electrode

        Parameters
        ----------
        x, y
            The spatial coordinate in millimeters.
        electrodes
            The electrode ID, or iterable of electrode IDs, if None, will
            return all electrodes
        Returns
        -------
        The electric potential for a stimulus current specified in attribute
            CURRENT_REFERENCE. If electrodes is a single value (not in an
            iterable) then the returned potential is also a single value or
            array, if multiple x, y are specified. Otherwise, the potentials are
            returned in a list.
        """
        if electrodes is None:
            electrodes = self.active_electrodes
        if not isinstance(electrodes, str):
            try:
                i = iter(electrodes)
            except TypeError:
                i = None
        else:
            i = None

        self.load(electrodes)
        if i is None:
            interpolator = self._funs[electrodes]
            return interpolator(x, y)
        else:
            out = []
            for e in i:
                interpolator = self._funs[e]
                out.append(interpolator(x, y))
            return out

    def get_potential_map(self, electrodes=None, ds=None):
        """
        Returns a gridded potential map for the entire extent

        Parameters
        ----------
        electrodes
            The electrode ID, or iterable of electrode IDs
        ds
            sample spacing, if unspecified, defaults to DS attribute value

        Returns
        -------
        potential, x coordinates, y coordinates

        """
        if ds is None:
            ds = self.DS
        if electrodes is None:
            electrodes = self.active_electrodes
        self.load(electrodes)

        xr = np.arange(*self.bounds['x'], ds)
        yr = np.arange(*self.bounds['y'], ds)
        X, Y = np.meshgrid(xr, yr)
        V = self.get_potential(X, Y, electrodes)
        if isinstance(V, list):
            V = [np.squeeze(v) for v in V]
        else:
            V = np.squeeze(V)
        return V, xr, yr

    @property
    def extent(self):
        """ Returns a bounding box for use with matplotlib's imshow """
        bounds = self.bounds
        return bounds['x'][0], bounds['x'][1], bounds['y'][0], bounds['y'][1]
    
    def get_uniform_points(self, num_pts : int):
        """ Generate uniformly distributed points within cuff 
        
        Parameters
        ----------
        num_pts : int
            The number of points
        
        Returns
        -------
        x, y : ndarray
            Flattened arrays of points
        """
        x, y = sampling.uniform_points_circle(num_pts, self.DIAMETER_MM)
        x += self._CX_MM
        y += self._CY_MM
        return x, y



class ElectrodeCuff4mm(GenericElectrodeCuff):
    """Loads FEM data for a cuff electrode with a 4mm diameter nerve."""
    DATA_DIR = COMSOL_DIR / "cuff_fem" / "4mm_diameter_round_cuff"
    FILEREG = re.compile(r"750umel_singleelon_1mAthrough_4mmnerve_el([0-9]+)")
    FILEGLOB = "*.txt"
    DIAMETER_MM = 4.0
    DS = 0.005  # Step size (mm)
    MAX_ELECTRODES = 16
    MAX_CURRENT_UA = 250

    def __str__(self):
        return "cuff_4m"


class FineArray(GenericElectrodeCuff):
    """Loads FEM data for a FINE array."""
    DATA_DIR = COMSOL_DIR / "fine"
    FILEREG = re.compile(r"1mAthrough_el([0-9]+)")
    FILEGLOB = "*.txt"
    DS = 0.005  # Step size (mm)
    DIAMETER_MM = None
    WIDTH_MM = 10
    HEIGHT_MM = 1.5
    MAX_ELECTRODES = 32
    CURRENT_REFERENCE_UA = 1000
    MAX_CURRENT_UA = 2000

    def __init__(self):
        super().__init__()
        self.num_base = len(self.files)

    def __str__(self):
        return "fine"
    
    @property
    def area(self):
        return self.WIDTH_MM*self.HEIGHT_MM
    
    @property
    def base_electrodes(self):
        return sorted(list(self.files.keys()))

    @property
    def electrodes(self):
        base_electrodes = self.base_electrodes
        flipped_electrodes = [e + self.num_base for e in base_electrodes]
        return base_electrodes + flipped_electrodes
    
    def load(self, electrodes=None):
        if electrodes is None:
            electrodes = self.base_electrodes
        elif isinstance(electrodes, int):
            electrodes %= self.num_base
        else:
            electrodes = [e % self.num_base for e in electrodes]
        super().load(electrodes)
    
    def _load_single(self, electrode):
        electrode %= self.num_base
        super()._load_single(electrode)
        if self._loaded.get(electrode + self.num_base, False):
            return
        # Generate the upside down electrode
        base_fun = self._funs[electrode]
        pts = base_fun.points.copy()
        pts[:,1] = -pts[:,1]
        vals = base_fun.values
        electrode += self.num_base
        self._funs[electrode] = interp.NearestNDInterpolator(pts, vals)
        self._loaded[electrode] = True

    def get_uniform_points(self, num_pts):
        """ Generate uniformly distributed points within array 
        
        Parameters
        ----------
        num_pts : int
            The number of points
        
        Returns
        -------
        x, y : ndarray
            Flattened arrays of points
        """
        x, y = sampling.grid_points_rectangle(
            num_pts, self.WIDTH_MM, self.HEIGHT_MM
        )
        x = x.ravel()
        y = y.ravel()
        x += self._CX_MM
        y += self._CY_MM
        return x, y


class UtahSlantElectrodeArray(GenericElectrodeCuff):
    DATA_DIR = COMSOL_DIR / "USEA"
    FILEREG = re.compile(r"VfromTerminal([0-9]+)")
    FILEGLOB = "*.gz"
    _CX_MM = 0.0    # millimeters
    _CY_MM = 0.0    # millimeters
    DS = 0.002  # step size (from inspection of data file)
    DATA_TO_MM = 1000
    CURRENT_REFERENCE_UA = 100     # uA
    MAX_CURRENT_UA = 200
    DIAMETER_MM = 3.0  # TODO: CHECK USEA REFERENCE DIAMETER!
    CFG = stimulus.NeuralStimulatorConfig(
        electrodes=stimulus._get_electrodes_within_nerve(
            nerve_radius=DIAMETER_MM/2),
        frequency_min=0,
        frequency_max=500,
        amplitude_min=0,
        amplitude_max=5,  # TODO: check what this should be.
        pulse_width_min=0,
        pulse_width_max=0.300,  # ms
        update_rate=33.0,  # Hz
        **stimulus._load_fem_for_usea(cx_mm=1.8, cy_mm=1.4,
                                      data_to_mm=DATA_TO_MM,
                                      nerve_radius=DIAMETER_MM/2),

    )
    MAX_ELECTRODES = len(CFG.electrodes)

    def __str__(self):
        return "usea"

    @property
    def bounds(self):
        if self._bounds is None:
            self._bounds = {
                'x': [min(self.CFG.reference_potential_xc),
                      max(self.CFG.reference_potential_xc)],
                'y': [min(self.CFG.reference_potential_yc),
                      max(self.CFG.reference_potential_yc)]
            }
        return self._bounds

    def _load_csv(self, filename):
        return pd.read_csv(self.DATA_DIR / filename)

    @property
    def electrodes(self):
        return sorted(self.CFG.electrodes)

    @property
    def num_electrodes(self):
        return len(self.CFG.electrodes)

    def _load_single(self, electrode):
        """
        Creates interpolation function with data for specified electrode

        Parameters
        ----------
        electrode
            A single electrode to load
        """

        if self._loaded[electrode]:
            return

        xy = np.vstack((self.CFG.reference_potential_xc,
                        self.CFG.reference_potential_yc))

        electrode_idx = int(np.where(
            electrode == np.asarray(self.electrodes))[0][0])
        self._funs[electrode] = interp.NearestNDInterpolator(
            xy.transpose(),
            self.CFG.reference_potential_map[electrode_idx])
        self._loaded[electrode] = True
        return

    def get_potential_map(self, electrodes=None, ds=None):
        """
        Returns a gridded potential map for the entire extent

        Parameters
        ----------
        electrodes
            The electrode ID, or iterable of electrode IDs
        ds
            sample spacing, if unspecified, defaults to DS attribute value

        Returns
        -------
        potential, x coordinates, y coordinates

        """
        if electrodes is None:
            electrodes = self.active_electrodes

        if ds is None:
            if isinstance(electrodes, int):
                map = self.CFG.reference_potential_map[
                electrodes==self.CFG.electrodes]
            else:
                map = []
                for idx, e in enumerate(electrodes):
                    map.append(self.CFG.reference_potential_map[idx])

            return map, self.CFG.reference_potential_xc, \
                   self.CFG.reference_potential_yc

        xr = np.arange(*self.bounds['x'], ds)
        yr = np.arange(*self.bounds['y'], ds)
        X, Y = np.meshgrid(xr, yr)
        V = self.get_potential(X, Y, electrodes)
        if isinstance(V, list):
            V = [np.squeeze(v) for v in V]
        else:
            V = np.squeeze(V)
        return V, xr, yr

