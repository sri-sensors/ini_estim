from .arrays import GenericElectrodeCuff, ElectrodeCuff4mm, \
    FineArray, UtahSlantElectrodeArray


electrode_arrays = dict(
    generic_cuff=GenericElectrodeCuff,
    cuff_4mm=ElectrodeCuff4mm,
    fine=FineArray,
    usea=UtahSlantElectrodeArray
)
