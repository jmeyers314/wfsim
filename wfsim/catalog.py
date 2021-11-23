import numpy as np
import lsst.sphgeom as sphgeom

# Add some missing functionality from sphgeom
# See LSSTDESC.Coord for definitions of these

def _dsq(pt1, pt2):
    return (
        (pt1.x()-pt2.x())**2
        + (pt1.y()-pt2.y())**2
        + (pt1.z()-pt2.z())**2
    )

def _cross(pt1, pt2):
    return (
        pt1.y() * pt2.z() - pt2.y() * pt1.z(),
        pt1.z() * pt2.x() - pt2.z() * pt1.x(),
        pt1.x() * pt2.y() - pt2.x() * pt1.y()
    )

def _triple(pt1, pt2, pt3):
    return np.linalg.det(
        [[ pt2.x(), pt2.y(), pt2.z() ],
         [ pt1.x(), pt1.y(), pt1.z() ],
         [ pt3.x(), pt3.y(), pt3.z() ]]
    )

def _alt_triple(pt1, pt2, pt3):
    dsq_AC = (pt1.x()-pt2.x())**2 + (pt1.y()-pt2.y())**2 + (pt1.z()-pt2.z())**2
    dsq_BC = (pt1.x()-pt3.x())**2 + (pt1.y()-pt3.y())**2 + (pt1.z()-pt3.z())**2
    dsq_AB = (pt3.x()-pt2.x())**2 + (pt3.y()-pt2.y())**2 + (pt3.z()-pt2.z())**2
    return 0.5 * (dsq_AC + dsq_BC - dsq_AB - 0.5 * dsq_AC * dsq_BC)

def distance_to(pt1, pt2):
    dsq = _dsq(pt1, pt2)
    if dsq < 3.99:
        return 2*np.arcsin(0.5 * np.sqrt(dsq))
    else:
        cx, cy, cz = _cross(pt1, pt2)
        crosssq = cx**2 + cy**2 + cz**2
        return np.pi - np.arcsin(np.sqrt(crosssq))

def angle_between(pt1, pt2, pt3):
    sinC = _triple(pt1, pt2, pt3)
    cosC = _alt_triple(pt1, pt2, pt3)

    C = np.arctan2(sinC, cosC)
    return C

def area(poly):
    vertices = poly.getVertices()
    s = 0
    N = len(vertices)
    for i in range(N):
        s += angle_between(
            vertices[(i+1)%N],
            vertices[i%N],
            vertices[(i+2)%N]
        )
    return np.abs(s) - (N-2)*np.pi


def _magfunc(m):
    """Approximate magnitude function between +4 and +25
    https://spacemath.gsfc.nasa.gov/stars/6Page103.pdf
    """
    return 10**(-0.0003*m**3 + 0.0019*m**2 + 0.484*m - 3.82)


def rotate(axis, angle, vec):
    ndim = vec.ndim
    vec = np.atleast_2d(vec)
    sth, cth = np.sin(angle), np.cos(angle)
    dot = np.dot(axis, vec.T)
    cross = np.cross(axis, vec)
    out = vec * cth
    out += cross * sth
    out += axis * dot[:, None] * (1-cth)
    if ndim == 1:
        out = out[0]
    return out


class MockStarCatalog:
    def __init__(self, level=7, seed=57721):
        self.htm = sphgeom.HtmPixelization(level)
        self.seed = seed
        self.bins = np.arange(10.0, 25.1, 0.1)
        self.bincounts = np.empty(len(self.bins)-1)
        for i, bin in enumerate(self.bins[:-1]):
            self.bincounts[i] = _magfunc(bin+0.1) - _magfunc(bin)
        self.density = np.sum(self.bincounts)  # per sq. degree

    def get_triangle_stars(self, idx):
        rng = np.random.default_rng(self.seed+idx)
        triangle = self.htm.triangle(idx)
        circle = triangle.getBoundingCircle()
        center = circle.getCenter()
        opening_angle = circle.getOpeningAngle().asRadians()
        area = circle.getArea() * (180/np.pi)**2
        N = rng.poisson(self.density*area)
        # uniformly sample cylinder then project to sphere
        zmin = np.cos(opening_angle)
        z = rng.uniform(zmin, 1, size=N)
        ph = rng.uniform(0, 2*np.pi, size=N)
        r = np.sqrt(1-z**2)
        x = r*np.cos(ph)
        y = r*np.sin(ph)
        # rotate to correct point on the sky
        axis = sphgeom.UnitVector3d.orthogonalTo(sphgeom.UnitVector3d.Z(), center)
        angle = np.pi/2 - sphgeom.LonLat.latitudeOf(center).asRadians()
        xyz = np.array([x, y, z]).T
        xyz = rotate(axis, angle, xyz)
        # sample magnitudes
        magbin = rng.choice(
            len(self.bincounts),
            size=N,
            p=self.bincounts/self.density
        )
        mag = self.bins[magbin] + rng.uniform(0, 0.1, size=N)
        # only keep points actually within triangle
        w = triangle.contains(*xyz.T)
        xyz = xyz[w]
        mag = mag[w]
        return xyz, mag

    def get_stars(self, polygon):
        ranges = self.htm.envelope(polygon)
        # For each spherical triangle, seed is index + self.seed
        # Uniformly populate spherical cap
        xyzs = []
        mags = []
        for begin, end in ranges:
            for idx in range(begin, end):
                xyz, mag = self.get_triangle_stars(idx)
                xyzs.append(xyz)
                mags.append(mag)
        # trim to polygon
        xyz = np.vstack(xyzs)
        mag = np.hstack(mags)
        w = polygon.contains(*xyz.T)
        xyz = xyz[w]
        mag = mag[w]
        return xyz, mag
