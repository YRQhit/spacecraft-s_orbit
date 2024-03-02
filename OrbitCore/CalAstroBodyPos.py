import numpy as np
from astropy.coordinates import get_body_barycentric, get_body, ICRS, GCRS ,get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
#
# #地球指向太阳的矢量
# def CalAstroBodyPos(JD):
#     # Convert Julian Date to Astropy Time object
#     time = Time(JD, format='jd', scale='utc')
#
#     # Calculate the position of Earth with respect to the Sun (barycentric coordinates)
#     earth_sun_vector = get_body_barycentric('earth', time)
#
#     # Convert the barycentric coordinates to geocentric coordinates (Earth-centered)
#     earth_geocentric_vector = -1 * earth_sun_vector.represent_as(CartesianRepresentation).xyz.to(u.km)
#
#     return np.array(earth_geocentric_vector.value)
#
#
# JD = 2459396.5  # Julian Date
# sun_earth_vector = CalAstroBodyPos(JD)
# print("Sun-Earth Vector (km):", sun_earth_vector)


# import numpy as np
# from scipy.optimize import newton
# from astropy.coordinates import get_body_barycentric, GCRS
# from astropy.time import Time
# def CalJDTDB(JD):
#     DAT = 36  # TAI-UTC, in seconds
#     JDTT = JD + (DAT + 32.184) / 24 / 3600
#     JDTDB = JDTT + (0.001657 * np.sin(6.24 + 0.017202 * (JDTT - 2451545.0))) / 24 / 3600
#     return JDTDB
#
# def kepler_E(e, M):
#     def kepler_eq(E):
#         return E - e * np.sin(E) - M
#
#     E0 = M + e / 2 if M < np.pi else M - e / 2
#     E = newton(kepler_eq, E0)
#     return E
#
# def CalOrbEle(JCTDB, OrbEle_0, dOrbEle_0):
#     OrbEle = OrbEle_0 + dOrbEle_0 * JCTDB
#     return OrbEle
#
# def CalPos(h, e, Omegaa, i, omegaa, thetaa, GM):
#     rp = (h**2 / GM) * (1 / (1 + e * np.cos(thetaa))) * (np.cos(thetaa) * np.array([1, 0, 0]) + np.sin(thetaa) * np.array([0, 1, 0]))
#     R3_Omegaa = np.array([[np.cos(Omegaa), np.sin(Omegaa), 0],
#                           [-np.sin(Omegaa), np.cos(Omegaa), 0],
#                           [0, 0, 1]])
#     R1_i = np.array([[1, 0, 0],
#                      [0, np.cos(i), np.sin(i)],
#                      [0, -np.sin(i), np.cos(i)]])
#     R3_omegaa = np.array([[np.cos(omegaa), np.sin(omegaa), 0],
#                           [-np.sin(omegaa), np.cos(omegaa), 0],
#                           [0, 0, 1]])
#     Q_pX = R3_Omegaa.T @ R1_i.T @ R3_omegaa.T
#     r = Q_pX @ rp
#     Pos = r.flatten()
#     return Pos
#
# def CalAstroBodyPosSCI(JCTDB, GM, a_0, da_0, e_0, de_0, i_0, di_0, Omegaa_0, dOmegaa_0, hat_omegaa_0, dhat_omegaa_0, L_0, dL_0):
#     a = CalOrbEle(JCTDB, a_0, da_0)
#     e = CalOrbEle(JCTDB, e_0, de_0)
#     i = CalOrbEle(JCTDB, i_0, di_0)
#     Omegaa = CalOrbEle(JCTDB, Omegaa_0, dOmegaa_0)
#     hat_omegaa = CalOrbEle(JCTDB, hat_omegaa_0, dhat_omegaa_0)
#     L = CalOrbEle(JCTDB, L_0, dL_0)
#     h = np.sqrt(GM * a * (1 - e**2))
#     omegaa = hat_omegaa - Omegaa
#     M = L - hat_omegaa
#     E = kepler_E(e, M)
#     thetaa = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
#     AstroBodyPosSCI = CalPos(h, e, Omegaa, i, omegaa, thetaa, GM)
#     return AstroBodyPosSCI
#
# def CalAstroBodyPos(JD):
#     d2r = np.pi / 180
#     JDTDB = CalJDTDB(JD)
#     JCTDB = (JDTDB - 2451545) / 36525
#
#     GM_s = 1.327122000000e+11  # Sun's gravitational constant, unit: km^3/s^2
#     EcliOblJ2000 = 23.43929 * d2r  # J2000.0 obliquity of the ecliptic, unit: rad
#     ECI2SCI = np.angle2dcm(EcliOblJ2000, 0, 0, 'XYZ')  # Transformation matrix from Earth-centered inertial coordinates to Solar-centered inertial coordinates
#
#     AU = 1.49597871e8
#     EarthPosSCI = CalAstroBodyPosSCI(JCTDB, GM_s, 1.00000011 * AU, -0.00000005 * AU, 0.01671022, -0.00003804, 0.00005 * d2r, -46.94 / 3600 * d2r, -11.26064 * d2r, -18228.25 / 3600 * d2r, 102.94719 * d2r, 1198.28 / 3600 * d2r, 100.46435 * d2r, 129597740.63 / 3600 * d2r)
#     SunPos = -ECI2SCI.T @ EarthPosSCI
#     return SunPos
#
# # Example usage:
# JD = 2459396.5  # Julian Date
# SunPos = CalAstroBodyPos(JD)
# print("Sun Position (km):", SunPos)

from scipy.optimize import newton
import numpy as np
import math
from astropy.coordinates import get_body_barycentric, GCRS
from astropy.time import Time
from astropy.coordinates.matrix_utilities import rotation_matrix
def CalJDTDB(JD):
    DAT = 36  # TAI-UTC, in seconds
    JDTT = JD + (DAT + 32.184) / 24 / 3600
    JDTDB = JDTT + (0.001657 * np.sin(6.24 + 0.017202 * (JDTT - 2451545.0))) / 24 / 3600
    return JDTDB

def kepler_E(e, M):
    def kepler_eq(E):
        return E - e * np.sin(E) - M

    E0 = M + e / 2 if M < np.pi else M - e / 2
    E = newton(kepler_eq, E0)
    return E

def CalOrbEle(JCTDB, OrbEle_0, dOrbEle_0):
    OrbEle = OrbEle_0 + dOrbEle_0 * JCTDB
    return OrbEle



def CalPos(h, e, Omegaa, i, omegaa, thetaa, GM):
    rp = (h ** 2 / GM) * (1 / (1 + e * np.cos(thetaa))) * np.array([np.cos(thetaa), np.sin(thetaa), 0])

    R3_Omegaa = np.array([[np.cos(Omegaa), np.sin(Omegaa), 0],
                          [-np.sin(Omegaa), np.cos(Omegaa), 0],
                          [0, 0, 1]])

    R1_i = np.array([[1, 0, 0],
                     [0, np.cos(i), np.sin(i)],
                     [0, -np.sin(i), np.cos(i)]])

    R3_omegaa = np.array([[np.cos(omegaa), np.sin(omegaa), 0],
                          [-np.sin(omegaa), np.cos(omegaa), 0],
                          [0, 0, 1]])

    Q_pX = np.dot(R3_Omegaa.T, np.dot(R1_i.T, R3_omegaa.T))
    r = np.dot(Q_pX, rp)
    r = np.reshape(r, (3, 1))
    Pos = r

    return Pos

def CalAstroBodyPosSCI(JCTDB, GM, a_0, da_0, e_0, de_0, i_0, di_0, Omegaa_0, dOmegaa_0, hat_omegaa_0, dhat_omegaa_0, L_0, dL_0):
    a = CalOrbEle(JCTDB, a_0, da_0)
    e = CalOrbEle(JCTDB, e_0, de_0)
    i = CalOrbEle(JCTDB, i_0, di_0)
    Omegaa = CalOrbEle(JCTDB, Omegaa_0, dOmegaa_0)
    hat_omegaa = CalOrbEle(JCTDB, hat_omegaa_0, dhat_omegaa_0)
    L = CalOrbEle(JCTDB, L_0, dL_0)
    h = np.sqrt(GM * a * (1 - e**2))
    omegaa = hat_omegaa_0 - Omegaa
    M = L - hat_omegaa
    E = kepler_E(e, M)
    thetaa = math.atan(math.tan(E / 2) * math.sqrt((1 + e) / (1 - e))) * 2
    # thetaa = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

    AstroBodyPosSCI = CalPos(h, e, Omegaa, i, omegaa, thetaa, GM)
    return AstroBodyPosSCI

def get_ECI_to_SCI_transform_matrix(obliquity):
    cos_ecl = np.cos(obliquity)
    sin_ecl = np.sin(obliquity)
    ECI_to_SCI_transform_matrix = np.array([[1, 0, 0],
                                            [0, cos_ecl, sin_ecl],
                                            [0, -sin_ecl, cos_ecl]])

    return ECI_to_SCI_transform_matrix

def CalAstroBodyPos(JD):
    d2r = np.pi / 180
    JDTDB = CalJDTDB(JD)
    JCTDB = (JDTDB - 2451545) / 36525

    GM_s = 1.327122000000e+11  # Sun's gravitational constant, unit: km^3/s^2
    EcliOblJ2000 = 23.43929 * d2r  # J2000.0 obliquity of the ecliptic, unit: rad

    # Calculate the transformation matrix from Earth-centered inertial (ECI) to Solar-centered inertial (SCI) coordinates
    ECI_to_SCI_transform_matrix = get_ECI_to_SCI_transform_matrix(EcliOblJ2000)

    AU = 1.49597871e8
    EarthPosSCI = CalAstroBodyPosSCI(JCTDB, GM_s, 1.00000011 * AU, -0.00000005 * AU, 0.01671022, -0.00003804, 0.00005 * d2r, -46.94 / 3600 * d2r, -11.26064 * d2r, -18228.25 / 3600 * d2r, 102.94719 * d2r, 1198.28 / 3600 * d2r, 100.46435 * d2r, 129597740.63 / 3600 * d2r)

    # Apply the transformation matrix to get the Sun's position in SCI coordinates
    SunPos = np.dot(ECI_to_SCI_transform_matrix.T, EarthPosSCI)

    return SunPos

# The CalJDTDB function and other necessary functions should be defined as well.

# Example usage:
JD = 2459396.5  # Julian Date
SunPos = CalAstroBodyPos(JD)
print("Sun Position (km):", SunPos)
