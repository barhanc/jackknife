"""Gentle snowfall driven by real time, independent of simulation speed."""
import math
import random
import time

import pygame


class Snowfall:
    def __init__(self, seed=481):
        rng = random.Random(seed)
        self.flakes = [(rng.random(), rng.random(), rng.uniform(.35,1), rng.uniform(0,math.tau)) for _ in range(135)]
        self.elapsed = 0.0
        self.last_time = time.perf_counter()

    def draw(self, surface, viewport):
        now = time.perf_counter()
        self.elapsed += min(.1,max(0,now-self.last_time))
        self.last_time = now
        for u,v,near,phase in self.flakes:
            x = viewport.left+(u*viewport.w+self.elapsed*(5+near*9)+math.sin(self.elapsed*.55+phase)*9)%max(1,viewport.w)
            y = viewport.top+(v*viewport.h+self.elapsed*(10+near*20))%max(1,viewport.h)
            color = (119,136,145) if near<.6 else (164,180,186) if near<.83 else (205,215,215)
            size = 1 if near<.6 else 2
            pygame.draw.rect(surface,color,(round(x),round(y),size,size))
