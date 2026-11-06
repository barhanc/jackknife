"""Seeded snow coverage with palette-based pixel shading, baked into the yard.

Shape uses noise at several scales rather than independent radial polygons:
https://www.redblobgames.com/maps/terrain-from-noise/
Separate surface detail and melting edges follow the approach described at:
https://www.sidefx.com/docs/houdini/nodes/sop/labs--snow_buildup-2.0.html
"""
import numpy as np
import pygame
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, shift


class SnowCover:
    RESOLUTION = 3  # World-anchored pixels, independent of route and viewport.

    def __init__(self, yard_w, yard_h, seed):
        rng = np.random.default_rng(seed)
        width, height = round(yard_w*self.RESOLUTION), round(yard_h*self.RESOLUTION)
        yy,xx = np.mgrid[:height,:width]

        def noise(spacing):
            rows,cols = max(3,round(yard_h/spacing)+2),max(3,round(yard_w/spacing)+2)
            grid = rng.standard_normal((rows,cols))
            coordinates = np.array([yy/(height-1)*(rows-2),xx/(width-1)*(cols-2)])
            values = map_coordinates(grid,coordinates,order=3,mode='reflect')
            return (values-values.mean())/max(values.std(),1e-6)

        broad,medium,fine = noise(9),noise(3),noise(.8)
        # Snow collects in connected pockets, particularly at the yard edges.
        edge_distance = np.minimum.reduce((xx,width-1-xx,yy,height-1-yy))/self.RESOLUTION
        edge_bias = .7*np.exp(-edge_distance/6)
        field = .78*broad+.27*medium+.07*fine+edge_bias-.97
        coverage = (field>0).astype(float)
        inside = np.asarray(distance_transform_edt(field>0))/self.RESOLUTION
        # A low, rounded drift; its rim melts smoothly into the pavement.
        elevation = .22*(1-np.exp(-inside/1.1))*(1+.09*fine)
        elevation = gaussian_filter(elevation,sigma=.7)
        dy,dx = np.gradient(elevation,1/self.RESOLUTION)
        normal_length = np.sqrt(1+dx*dx+dy*dy)
        light = np.clip((.45*dx+.65*dy+.8)/normal_length,0,1.2)
        interior = 1-np.exp(-inside/.5)
        # Four deliberate colours and subtle ordered dithering keep the snow
        # in the same visual language as the arcade HUD and vehicle sprites.
        bayer = np.array([[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]])
        dither = (bayer[yy%4,xx%4]/15-.5)*4
        brightness = 125+34*interior+90*(light-.8)+dither
        palette = np.array([(76,94,104),(112,131,140),(153,169,175),(180,191,193)])
        rgb = palette[np.digitize(brightness,[132,151,162])]
        self.coverage = coverage
        self.elevation = elevation
        self.snow = self._surface(rgb,coverage*255)
        # A single pixel of cast shadow replaces the blurred dark outline.
        shadow = shift(coverage,(1,1),order=0,mode='constant')
        shadow_rgb = np.zeros((height,width,3));shadow_rgb[:]=(6,12,18)
        self.shadow = self._surface(shadow_rgb,shadow*70)
        self._size = None
        self._scaled = None
        self.yard_w,self.yard_h = yard_w,yard_h

    @staticmethod
    def _surface(rgb,alpha):
        rgba = np.dstack((np.clip(rgb,0,255),np.clip(alpha,0,255))).astype(np.uint8)
        height,width = alpha.shape
        return pygame.image.frombytes(rgba.tobytes(),(width,height),'RGBA')

    def draw(self, surface, camera):
        size = (max(1,round(self.yard_w*camera.scale)),max(1,round(self.yard_h*camera.scale)))
        if self._size != size:
            self._size = size
            self._scaled = (pygame.transform.scale(self.shadow,size),pygame.transform.scale(self.snow,size))
        assert self._scaled is not None
        position = camera.project(-self.yard_w/2,self.yard_h/2)
        for layer in self._scaled:
            surface.blit(layer,position)
