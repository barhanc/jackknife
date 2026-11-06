"""Small cached light fields; rotation and placement follow the vehicle bodies."""
import math

import numpy as np
import pygame


class TruckLighting:
    def __init__(self):
        self._scale = None
        self._sprites = {}

    @staticmethod
    def headlights(state, dims):
        c, s = math.cos(state.theta0), math.sin(state.theta0)
        return [(state.x0+4.45*c-side*s, state.y0+4.45*s+side*c)
                for side in (-dims.cab_width*.72*.46, dims.cab_width*.72*.46)]

    @staticmethod
    def taillights(state, dims, rear_axle):
        x, y, heading = rear_axle
        c, s = math.cos(heading), math.sin(heading)
        offset = dims.L2+.85-7.7-.08
        return [(x+offset*c-side*s, y+offset*s+side*c)
                for side in (-dims.trailer2_width*.44, dims.trailer2_width*.44)]

    @staticmethod
    def _image(rgb, alpha):
        height,width = alpha.shape
        image = pygame.Surface((width,height),pygame.SRCALPHA,32)
        colors = pygame.surfarray.pixels3d(image)
        colors[:] = np.asarray(rgb,dtype=np.uint8).transpose(1,0,2)
        del colors
        opacity = pygame.surfarray.pixels_alpha(image)
        opacity[:] = np.asarray(alpha,dtype=np.uint8).T
        del opacity
        return image

    def _beam(self, scale, length, strength):
        margin = .8
        width = max(4,round((length+2*margin)*scale))
        height = max(4,round((length*.5+2)*scale))
        yy,xx = np.mgrid[:height,:width]
        origin = (round(margin*scale),height//2)
        x,y = (xx-origin[0])/scale,(yy-origin[1])/scale
        distance = np.maximum(x,0)
        spread = .2+.22*distance
        outer = np.exp(-1.8*(y/spread)**2)
        core = np.exp(-1.8*(y/(.12+.09*distance))**2)
        forward = 1/(1+np.exp(-np.clip(x*10,-40,40)))
        cutoff = np.clip((length-x)/2,0,1)**2
        intensity = (.7*outer+.3*core)*np.exp(-distance/6)*forward*cutoff
        alpha = np.clip(strength*intensity,0,255)
        warm = core*np.exp(-distance/7)
        rgb = np.stack((205+45*warm,222+20*warm,235-24*warm),axis=-1)
        return self._image(rgb,alpha),origin

    def _halo(self, scale, color, radius, strength):
        size = max(5,2*round(radius*scale)+1)
        yy,xx = np.mgrid[:size,:size]
        center = size//2
        r2 = ((xx-center)/(radius*scale))**2+((yy-center)/(radius*scale))**2
        alpha = strength*np.exp(-4*r2)*np.clip(1-r2,0,1)
        rgb = np.empty((size,size,3),dtype=np.uint8);rgb[:]=color
        return self._image(rgb,alpha),(center,center)

    def _prepare(self, scale):
        # Quantizing the cache key avoids rebuilding during ordinary animation.
        key = round(scale,3)
        if key == self._scale:
            return
        self._scale = key
        self._sprites = {
            'head': self._beam(scale,13.5,72),
            'reverse': self._beam(scale,6.5,40),
            'white': self._halo(scale,(255,239,199),1.3,65),
            'red': self._halo(scale,(255,48,58),1.25,75),
        }

    @staticmethod
    def _blit(surface, sprite, position, heading=0.0):
        image,origin = sprite
        angle = math.degrees(heading)
        rotated = pygame.transform.rotate(image,angle) if abs(angle)>.01 else image
        center = image.get_rect().center
        dx,dy = origin[0]-center[0],origin[1]-center[1]
        c,s = math.cos(heading),math.sin(heading)
        offset = (c*dx+s*dy,-s*dx+c*dy)
        rect = rotated.get_rect(center=(round(position[0]-offset[0]),round(position[1]-offset[1])))
        surface.blit(rotated,rect)

    def draw_ground(self, surface, camera, state, dims, rear_axle, reversing):
        self._prepare(camera.scale)
        for point in self.headlights(state,dims):
            pixel = camera.project(*point)
            self._blit(surface,self._sprites['head'],pixel,state.theta0)
            self._blit(surface,self._sprites['white'],pixel)
        for point in self.taillights(state,dims,rear_axle):
            self._blit(surface,self._sprites['red'],camera.project(*point))
        if reversing:
            x,y,heading = rear_axle
            c,s = math.cos(heading),math.sin(heading)
            offset = dims.L2+.85-7.7-.08
            for side in (-dims.trailer2_width*.27,dims.trailer2_width*.27):
                pixel = camera.project(x+offset*c-side*s,y+offset*s+side*c)
                self._blit(surface,self._sprites['reverse'],pixel,heading+math.pi)

    @staticmethod
    def draw_lens(surface, camera, point, heading, color, height=0.0):
        """Recessed bezel, bevel and bright inset instead of a flat colored dot."""
        cx,cy = camera.project(*point)
        cy -= round(height*camera.scale)
        c,s = math.cos(heading),math.sin(heading)
        def box(length,width):
            hl,hw=length*camera.scale/2,width*camera.scale/2
            return [(round(cx+a*c+b*s),round(cy-a*s+b*c)) for a,b in [(-hl,-hw),(hl,-hw),(hl,hw),(-hl,hw)]]
        body = box(.32,.4)
        pygame.draw.polygon(surface,(7,11,17),[(x+1,y+1) for x,y in body])
        pygame.draw.polygon(surface,(100,111,120),body)
        pygame.draw.polygon(surface,(21,29,37),box(.25,.32))
        pygame.draw.polygon(surface,color,box(.14,.24))
        pygame.draw.circle(surface,tuple(min(255,c+55) for c in color),(cx,cy),max(1,round(camera.scale*.055)))
