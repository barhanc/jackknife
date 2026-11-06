"""Deterministic depot scenery and route markings, baked when the route changes."""
from dataclasses import dataclass
import math

import numpy as np
import pygame

from .snow import SnowCover

ROAD_PAINT = (222, 228, 226)
WORN_PAINT = (179, 188, 186)

@dataclass(frozen=True)
class DepotProp:
    kind: str
    x: float
    y: float
    length: float
    width: float
    heading: float = 0.0
    number: int = 0


@dataclass(frozen=True)
class LoadingBay:
    x: float
    y: float
    heading: float
    warehouse: bool


def box_points(x, y, length, width, heading=0.0):
    c, s = math.cos(heading), math.sin(heading)
    return [(x+a*c-b*s, y+a*s+b*c) for a, b in [
        (-length/2, -width/2), (length/2, -width/2),
        (length/2, width/2), (-length/2, width/2),
    ]]


class DepotTerrain:
    ROAD_HALF_WIDTH = 4.8
    MASK_SCALE = 2.0

    def __init__(self, yard_w=160.0, yard_h=100.0, seed=2718):
        self.yard_w, self.yard_h, self.seed = yard_w, yard_h, seed
        self.mask_size = (round(yard_w*self.MASK_SCALE), round(yard_h*self.MASK_SCALE))
        self.path = None
        self.props = []
        self.dock = None
        self.reserved_mask = pygame.Mask(self.mask_size)
        rng = np.random.default_rng(seed)
        self.slabs = []
        for ix, x in enumerate(np.arange(-yard_w/2, yard_w/2, 16.0)):
            for iy, y in enumerate(np.arange(-yard_h/2, yard_h/2, 10.0)):
                tone = int(rng.integers(-3, 4))
                self.slabs.append((x, y, tuple(c+tone for c in (34, 38, 43))))
        self.grain = [(float(rng.uniform(-yard_w/2, yard_w/2)), float(rng.uniform(-yard_h/2, yard_h/2)), int(rng.integers(27, 46))) for _ in range(1500)]
        self.patches = [(float(rng.uniform(-73, 73)), float(rng.uniform(-44, 44)), float(rng.uniform(2, 6)), float(rng.uniform(1, 3)), float(rng.uniform(-.2, .2))) for _ in range(28)]
        self.stains = [(float(rng.uniform(-72, 72)), float(rng.uniform(-42, 42)), float(rng.uniform(.3, 1.1))) for _ in range(24)]
        self.snow = SnowCover(yard_w,yard_h,seed+91)

    def invalidate(self):
        self.path = None

    def _mask_point(self, point):
        return round((point[0]+self.yard_w/2)*self.MASK_SCALE), round((self.yard_h/2-point[1])*self.MASK_SCALE)

    def footprint_mask(self, prop, margin=0.0):
        surface = pygame.Surface(self.mask_size, depth=32)
        points = box_points(prop.x, prop.y, prop.length+2*margin, prop.width+2*margin, prop.heading)
        pygame.draw.polygon(surface, (255,255,255), [self._mask_point(p) for p in points])
        return pygame.mask.from_threshold(surface, (255,255,255), (1,1,1,255))

    def _inside_yard(self, prop, margin=0.0):
        return all(abs(x) <= self.yard_w/2-1 and abs(y) <= self.yard_h/2-1 for x, y in box_points(prop.x, prop.y, prop.length+2*margin, prop.width+2*margin, prop.heading))

    @staticmethod
    def _capsules(size, points, radius):
        surface = pygame.Surface(size, depth=32)
        radius = max(1, round(radius))
        if len(points) > 1:
            pygame.draw.lines(surface, (255,255,255), False, points, 2*radius)
        for point in points:
            pygame.draw.circle(surface, (255,255,255), point, radius)
        return pygame.mask.from_threshold(surface, (255,255,255), (1,1,1,255))

    def configure(self, path, kinematics, starting_state):
        if self.path is path:
            return
        self.path = path
        self.props = []
        self.dock = None
        if path is None or len(path.x) < 2:
            self.reserved_mask = pygame.Mask(self.mask_size)
            return
        dims = kinematics.dims
        # Protect even strongly articulated bodies while the tracking axle is
        # within the 5.5 m gameplay limit, plus the initial aligned footprint.
        reach = dims.L0+dims.L1+dims.L2+dims.cab_front_overhang
        clearance = math.hypot(reach, max(dims.cab_width,dims.trailer1_width,dims.trailer2_width)/2)+6.0
        route = [self._mask_point(p) for p in zip(path.x, path.y)]
        self.reserved_mask = self._capsules(self.mask_size, route, clearance*self.MASK_SCALE)
        self.starting_mask = pygame.Mask(self.mask_size)
        for key in ("cab", "trailer1", "trailer2"):
            body = kinematics.get_body_polygons(starting_state)[key]
            points = [self._mask_point(p) for p in body]
            extra = self._capsules(self.mask_size, points+[points[0]], 3*self.MASK_SCALE)
            polygon = pygame.Surface(self.mask_size, depth=32)
            pygame.draw.polygon(polygon, (255,255,255), points)
            extra.draw(pygame.mask.from_threshold(polygon,(255,255,255),(1,1,1,255)), (0,0))
            self.reserved_mask.draw(extra, (0,0))
            self.starting_mask.draw(extra, (0,0))
        ex, ey, heading = float(path.x[-1]), float(path.y[-1]), float(path.headings[-1])
        dock_reserve = DepotProp("bay", ex-5*math.cos(heading), ey-5*math.sin(heading), 18, 14, heading)
        self.reserved_mask.draw(self.footprint_mask(dock_reserve), (0,0))
        near_edge = min(self.yard_w/2-abs(ex), self.yard_h/2-abs(ey)) < 12
        building = DepotProp("warehouse", ex+5*math.cos(heading), ey+5*math.sin(heading), 4, 14, heading, 1)
        building_mask = self.footprint_mask(building, margin=1.0)
        early_points = [self._mask_point((x,y)) for x,y,distance in zip(path.x,path.y,path.s) if distance < path.total_length-12]
        early_route = self._capsules(self.mask_size, early_points, 6*self.MASK_SCALE)
        warehouse = (near_edge and self._inside_yard(building)
                     and early_route.overlap(building_mask,(0,0)) is None
                     and self.starting_mask.overlap(building_mask,(0,0)) is None)
        self.dock = LoadingBay(ex, ey, heading, warehouse)
        occupied = self.reserved_mask.copy()
        candidates = []
        for y in (-43.0, 43.0):
            for x in (-52.0, 0.0, 52.0):
                candidates.append(DepotProp("warehouse",x,y,32,10,0,len(candidates)+1))
        for x in (-73.0, 73.0):
            for y in (-31.0,-13.0,5.0,23.0):
                candidates.append(DepotProp("container",x,y,12,3,math.pi/2,len(candidates)+1))
        for y in (-30.0, 30.0):
            for x in (-56.0,-28.0,0.0,28.0,56.0):
                candidates.append(DepotProp("trailer",x,y,9,2.6,0,len(candidates)+1))
        for x in (-76.0,-38.0,0.0,38.0,76.0):
            for y in (-46.0,46.0):
                candidates.append(DepotProp("light",x,y,2,2,0,len(candidates)+1))
        for prop in candidates:
            footprint = self.footprint_mask(prop, margin=1.0)
            if self._inside_yard(prop, margin=1.0) and occupied.overlap(footprint,(0,0)) is None:
                self.props.append(prop)
                occupied.draw(footprint,(0,0))

    @staticmethod
    def _polygon(surface, camera, points, color, width=0):
        pygame.draw.polygon(surface, color, [camera.project(*p) for p in points], width)

    @staticmethod
    def _label(surface, camera, text, point, font, color, heading=0.0):
        if camera.scale < 3.0:
            return
        label = font.render(text, False, color)
        ink = label.get_bounding_rect()
        if ink.w and ink.h:
            label = label.subsurface(ink)
        angle = (math.degrees(heading)+90)%180-90
        if abs(angle) > 3:
            label = pygame.transform.rotate(label, angle)
        surface.blit(label, label.get_rect(center=camera.project(*point)))

    def draw_ground(self, surface, camera, font, editing=False):
        surface.fill((11,15,20))
        for x,y,color in self.slabs:
            self._polygon(surface,camera,box_points(x+8,y+5,16,10),color)
            self._polygon(surface,camera,box_points(x+8,y+5,16,10),(24,29,34),1)
        for x,y,length,width,heading in self.patches:
            points = box_points(x,y,length,width,heading)
            self._polygon(surface,camera,points,(28,33,38))
            self._polygon(surface,camera,points,(40,44,47),1)
        for x,y,tone in self.grain:
            pixel = camera.project(x,y)
            if surface.get_rect().collidepoint(pixel):
                surface.set_at(pixel, (tone,tone+3,tone+5))
        for x,y,radius in self.stains:
            p = camera.project(x,y)
            pygame.draw.circle(surface,(26,30,34),p,max(1,round(radius*camera.scale)))
        if not editing:
            self.snow.draw(surface,camera)
        for y in (-self.yard_h/2+3, self.yard_h/2-3):
            left,right=camera.project(-self.yard_w/2+5,y),camera.project(self.yard_w/2-5,y)
            pygame.draw.line(surface,(18,23,28),left,right,max(2,round(camera.scale*.35)))
            for x in np.arange(-self.yard_w/2+5,self.yard_w/2-5,1.2):
                pygame.draw.line(surface,(52,57,59),camera.project(x,y-.16),camera.project(x,y+.16),1)
        perimeter = box_points(0,0,self.yard_w,self.yard_h)
        self._polygon(surface,camera,perimeter,(75,82,83),max(2,round(camera.scale*.45)))
        self._polygon(surface,camera,box_points(0,0,self.yard_w-2,self.yard_h-2),(53,63,68),1)
        for y in np.arange(-self.yard_h/2+2,self.yard_h/2,8):
            for x in (-self.yard_w/2,self.yard_w/2):
                pygame.draw.circle(surface,(97,105,106),camera.project(x,y),max(1,round(camera.scale*.2)))
        self._label(surface,camera,"NORTH APRON // 04",(0,35),font,(76,81,82))
        self._label(surface,camera,"FREIGHT ONLY",(-52,-35),font,(76,81,82))
        if not editing:
            lights = pygame.Surface(surface.get_size(),pygame.SRCALPHA,32)
            for prop in self.props:
                if prop.kind == "light":
                    pygame.draw.circle(lights,(236,180,83,8),camera.project(prop.x,prop.y),max(2,round(camera.scale*6)))
                    pygame.draw.circle(lights,(236,180,83,12),camera.project(prop.x,prop.y),max(2,round(camera.scale*3)))
            surface.blit(lights,(0,0))
            for prop in self.props:
                self._draw_prop(surface,camera,font,prop)

    def _draw_prop(self, surface, camera, font, prop):
        points = [camera.project(*p) for p in box_points(prop.x,prop.y,prop.length,prop.width,prop.heading)]
        shadow = [(x+max(2,round(camera.scale*.45)),y+max(3,round(camera.scale*.7))) for x,y in points]
        pygame.draw.polygon(surface,(16,20,26),shadow)
        if prop.kind == "light":
            center = camera.project(prop.x,prop.y)
            pygame.draw.circle(surface,(58,62,62),center,max(2,round(camera.scale*.6)))
            pygame.draw.line(surface,(136,136,117),center,(center[0]-2,center[1]-round(camera.scale*1.3)),2)
            pygame.draw.circle(surface,(255,207,117),(center[0]-2,center[1]-round(camera.scale*1.3)),max(2,round(camera.scale*.22)))
            return
        palette = [(57,77,83),(91,78,62),(62,76,67)]
        roof_color = (48,53,61) if prop.kind == "warehouse" else palette[prop.number%len(palette)]
        if prop.kind == "trailer":
            c,s = math.cos(prop.heading),math.sin(prop.heading)
            for axle in (-prop.length*.3-.45,-prop.length*.3+.45):
                for side in (-prop.width*.54,prop.width*.54):
                    wheel = box_points(prop.x+axle*c-side*s,prop.y+axle*s+side*c,.8,.35,prop.heading)
                    self._polygon(surface,camera,wheel,(10,14,18))
            nose = box_points(prop.x+(prop.length/2+.25)*c,prop.y+(prop.length/2+.25)*s,.5,prop.width*.6,prop.heading)
            self._polygon(surface,camera,nose,(40,47,54))
        lift = max(2,round(camera.scale*(.65 if prop.kind == "warehouse" else .35)))
        top = [(x,y-lift) for x,y in points]
        pygame.draw.polygon(surface,(26,33,40),points)
        for i in range(4):
            j=(i+1)%4
            pygame.draw.polygon(surface,(36,42,49),[points[i],points[j],top[j],top[i]])
        pygame.draw.polygon(surface,roof_color,top)
        pygame.draw.polygon(surface,tuple(c+20 for c in roof_color),top,1)
        c,s=math.cos(prop.heading),math.sin(prop.heading)
        for off in np.arange(-prop.length/2+.5,prop.length/2,.7 if prop.kind == "container" else 2):
            endpoints=[camera.project(prop.x+off*c-side*s,prop.y+off*s+side*c) for side in (-prop.width*.42,prop.width*.42)]
            pygame.draw.line(surface,tuple(v+9 for v in roof_color),(endpoints[0][0],endpoints[0][1]-lift),(endpoints[1][0],endpoints[1][1]-lift),1)
        if prop.kind == "warehouse":
            # Loading shutters sit on the visible facade, below the raised roof.
            south_edge = sorted(points, key=lambda p:p[1], reverse=True)[:2]
            left,right = sorted(south_edge)
            for off in ((-9,0,9) if prop.length >= 24 else (0,)):
                fraction = .5+off/prop.length
                center_x = round(left[0]+fraction*(right[0]-left[0]))
                center_y = round(left[1]+fraction*(right[1]-left[1]))
                width = max(3,round(camera.scale*3.6))
                door=pygame.Rect(center_x-width//2,center_y-lift,width,lift)
                pygame.draw.rect(surface,(19,25,31),door)
                pygame.draw.rect(surface,(126,115,74),door,1)
            self._label(surface,camera,f"DEPOT {prop.number:02d}",(prop.x,prop.y+2.7),font,(158,160,145))
        else:
            self._label(surface,camera,f"{prop.number:02d}",(prop.x,prop.y),font,(166,174,166),prop.heading)

    def route_mask(self, camera, path, half_width):
        points = [camera.project(x,y) for x,y in zip(path.x,path.y)]
        return self._capsules((camera.viewport.right,camera.viewport.bottom),points,half_width*camera.scale)

    @staticmethod
    def _paint_mask(surface, mask, color):
        layer = pygame.Surface(mask.get_size(),pygame.SRCALPHA,32)
        mask.to_surface(layer,setcolor=(*color,255),unsetcolor=(0,0,0,0))
        surface.blit(layer,(0,0))

    def draw_route(self, surface, camera, path, font):
        if path is None or len(path.x)<2:
            return
        shoulder = self.route_mask(camera,path,5.3)
        road = self.route_mask(camera,path,self.ROAD_HALF_WIDTH)
        inner = self.route_mask(camera,path,self.ROAD_HALF_WIDTH-.24)
        self._paint_mask(surface,shoulder,(44,47,47))
        self._paint_mask(surface,road,(19,26,32))
        edge = road.copy();edge.erase(inner,(0,0))
        self._paint_mask(surface,edge,ROAD_PAINT)
        self.road_mask, self.edge_mask = road, edge
        for x,y,tone in self.grain:
            pixel = camera.project(x,y)
            if 0<=pixel[0]<road.get_size()[0] and 0<=pixel[1]<road.get_size()[1] and road.get_at(pixel):
                surface.set_at(pixel,(25,32,38))
        for side in (-1.25,1.25):
            points=[camera.project(x-math.sin(th)*side,y+math.cos(th)*side) for x,y,th in zip(path.x,path.y,path.headings)]
            pygame.draw.lines(surface,(14,20,25),False,points,max(1,round(camera.scale*.28)))
        for i in range(len(path.x)-1):
            x,y,th=path.x[i],path.y[i],path.headings[i]
            if int(path.s[i]/2.4)%2 == 0:
                pygame.draw.line(surface,ROAD_PAINT,camera.project(x,y),camera.project(path.x[i+1],path.y[i+1]),max(1,round(camera.scale*.15)))
            if abs(path.curvatures[i])>.055 and i%3 == 0:
                side=-math.copysign(5.05,path.curvatures[i])
                cx,cy=x-math.sin(th)*side,y+math.cos(th)*side
                pixel=camera.project(cx,cy)
                if 0<=pixel[0]<road.get_size()[0] and 0<=pixel[1]<road.get_size()[1] and not road.get_at(pixel):
                    self._polygon(surface,camera,box_points(cx,cy,1.2,.38,th),(196,156,64))
        for distance in np.arange(13,path.total_length-10,28):
            i=min(len(path.x)-1,int(np.searchsorted(path.s,distance)))
            x,y,th=path.x[i],path.y[i],path.headings[i]
            c,s=math.cos(th),math.sin(th)
            arrow=[(x+1.2*c,y+1.2*s),(x-.8*c-.7*s,y-.8*s+.7*c),(x-.3*c,y-.3*s),(x-.8*c+.7*s,y-.8*s-.7*c)]
            self._polygon(surface,camera,arrow,ROAD_PAINT)
        self._draw_dock(surface,camera,font)

    def _draw_dock(self, surface, camera, font):
        bay=self.dock
        if bay is None:
            return
        c,s=math.cos(bay.heading),math.sin(bay.heading)
        def point(a,b):return bay.x+a*c-b*s,bay.y+a*s+b*c
        frame=[point(-11,-4.5),point(0,-4.5),point(0,4.5),point(-11,4.5)]
        # Painted apron markings, rather than a physical wall across the route.
        for start,end in [(frame[0],frame[1]),(frame[1],frame[2]),(frame[2],frame[3])]:
            pygame.draw.line(surface,ROAD_PAINT,camera.project(*start),camera.project(*end),max(1,round(camera.scale*.2)))
        for side in (-1,1):
            for a in np.arange(-10,-1,2):
                pygame.draw.line(surface,WORN_PAINT,camera.project(*point(a,side*4.5)),camera.project(*point(a+1,side*3.8)),max(1,round(camera.scale*.14)))
            bumper=[point(1.7,side*3.1),point(1.7,side*4.2)]
            pygame.draw.line(surface,(10,14,18),camera.project(*bumper[0]),camera.project(*bumper[1]),max(2,round(camera.scale*.5)))
        self._label(surface,camera,"BAY 01",point(-5.5,5.8),font,ROAD_PAINT,bay.heading)
        if bay.warehouse:
            building=DepotProp("warehouse",*point(5,0),4,14,bay.heading,1)
            self._draw_prop(surface,camera,font,building)

    def draw_dock_lights(self, surface, camera, arrived, glow=False):
        if self.dock is None:
            return
        bay=self.dock
        color=(105,235,157) if arrived else (245,113,68)
        for side in (-4.7,4.7):
            p=camera.project(bay.x-math.sin(bay.heading)*side,bay.y+math.cos(bay.heading)*side)
            radius=max(3,round(camera.scale*.45))
            if glow:
                pygame.draw.circle(surface,(*color,35),p,radius*3)
            else:
                pygame.draw.rect(surface,(12,17,22),(p[0]-radius-2,p[1]-radius-2,2*radius+4,2*radius+4))
                pygame.draw.rect(surface,color,(p[0]-radius,p[1]-radius,2*radius+1,2*radius+1))
