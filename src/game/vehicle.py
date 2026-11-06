"""A polygon miniature: real body height, with the driving model kept in 2D."""
import math

import pygame


class VehicleRenderer:
    """Project simple solids and paint their visible faces from far to near."""

    LEAD_LENGTH = 4.1
    LEAD_CENTER = -2.4
    LEAD_NOSE_CHAMFER = .8

    @staticmethod
    def body_corners(length, width, nose_chamfer=0.0):
        hl,hw=length/2,width/2
        if nose_chamfer:
            return [(-hl,-hw),(hl-nose_chamfer,-hw),(hl,-hw+nose_chamfer),
                    (hl,hw-nose_chamfer),(hl-nose_chamfer,hw),(-hl,hw)]
        return [(-hl,-hw),(hl,-hw),(hl,hw),(-hl,hw)]

    def draw_shadows(self, surface, camera, state, dims, kinematics):
        x0,y0,h0=state.x0,state.y0,state.theta0
        x1,y1,h1=kinematics.get_trailer1_axle(state)
        x2,y2,h2=kinematics.get_trailer2_axle(state)
        bodies = [(x0,y0,h0,1.9,5.3,dims.cab_width),
                  (x0,y0,h1,self.LEAD_CENTER,self.LEAD_LENGTH,dims.trailer1_width),
                  (x2,y2,h2,dims.L2+.85-7.7/2,7.7,dims.trailer2_width)]
        # Broad faint penumbra, then progressively darker, tighter silhouettes.
        for spread,alpha in ((.65,16),(.42,28),(.23,45),(0,76)):
            for x,y,h,offset,length,width in bodies:
                c,s=math.cos(h),math.sin(h)
                cx,cy=x+offset*c+.55,y+offset*s-.75
                points=[camera.project(cx+a*c-b*s,cy+a*s+b*c)
                        for a,b in [(-length/2-spread,-width/2-spread),
                                    (length/2+spread,-width/2-spread),
                                    (length/2+spread,width/2+spread),
                                    (-length/2-spread,width/2+spread)]]
                pygame.draw.polygon(surface,(4,8,13,alpha),points)
        for x,y,h,width in ((x0,y0,h0,dims.cab_width),(x1,y1,h1,dims.trailer1_width),(x2,y2,h2,dims.trailer2_width)):
            c,s=math.cos(h),math.sin(h)
            for offset in (-.58,.58):
                for side in (-width*.47,width*.47):
                    p=camera.project(x+offset*c-side*s,y+offset*s+side*c)
                    rect=pygame.Rect(0,0,max(3,round(camera.scale*1.05)),max(2,round(camera.scale*.55)))
                    rect.center=p
                    pygame.draw.ellipse(surface,(3,6,10,145),rect)

    def draw(self, surface, camera, state, dims, kinematics):
        faces = []

        def project(point):
            x, y, z = point
            # A fixed oblique height axis exposes both side and end faces.
            return camera.project(x - .10*z, y + .30*z)

        def depth(points):
            return sum(.10*x - .30*y + z for x, y, z in points)/len(points)

        def polygon(points, color, outline=None):
            faces.append((depth(points), points, color, outline, 0))

        def line(points, color, width=1):
            faces.append((depth(points)+.002, points, color, None, width))

        def point(cx, cy, heading, x, y, z):
            c, s = math.cos(heading), math.sin(heading)
            return (cx+x*c-y*s, cy+x*s+y*c, z)

        def box(cx, cy, heading, length, width, bottom, top, color, detailing=False, nose_chamfer=0.0):
            def p(x, y, z):
                return point(cx, cy, heading, x, y, z)
            corners = self.body_corners(length,width,nose_chamfer)
            bevel = min(.14, (top-bottom)*.2, width*.08)
            shoulder = top-bevel
            inset = [(x-math.copysign(bevel,x),y-math.copysign(bevel,y)) for x,y in corners]
            roof = [p(x,y,top) for x,y in inset]
            roof_color = tuple(min(255,round(c*1.08+8)) for c in color)
            polygon(roof, roof_color)
            for i,(ax,ay) in enumerate(corners):
                j=(i+1)%len(corners)
                bx,by=corners[j]
                ix,iy=inset[i];jx,jy=inset[j]
                polygon([p(ax,ay,shoulder),p(bx,by,shoulder),p(jx,jy,top),p(ix,iy,top)],
                        tuple(min(255,round(v*1.02+4)) for v in color))
            for i, (ax,ay) in enumerate(corners):
                bx,by = corners[(i+1)%len(corners)]
                # Outward wall normal in world space; hidden walls are omitted.
                nx,ny = by-ay,ax-bx
                c,s = math.cos(heading),math.sin(heading)
                nx,ny = nx*c-ny*s,nx*s+ny*c
                if nx*.10-ny*.30 <= 0:
                    continue
                normal_length = math.hypot(nx,ny)
                light = max(0,(-.6*nx-.8*ny)/normal_length)
                shade = .72+.20*light
                wall = tuple(round(v*shade) for v in color)
                polygon([p(ax,ay,bottom),p(bx,by,bottom),p(bx,by,shoulder),p(ax,ay,shoulder)],wall,tuple(round(v*.8) for v in wall))
                if detailing:
                    # Framing and ribs belong to the wall plane, not the roof.
                    for z in (bottom+.12,top-.12):
                        line([p(ax,ay,z),p(bx,by,z)],(109,139,145))
                    if abs(ax-bx) > length*.8:
                        for j in range(1,max(2,round(length/.6))):
                            t=j/max(2,round(length/.6))
                            x,y=ax+(bx-ax)*t,ay+(by-ay)*t
                            line([p(x,y,bottom+.2),p(x,y,top-.2)],tuple(min(255,v+9) for v in wall))
                        # White fleet stripe and amber side markers.
                        line([p(ax,ay,bottom+.48),p(bx,by,bottom+.48)],(195,205,194),max(1,round(camera.scale*.09)))
                        for t in (.08,.5,.92):
                            x,y=ax+(bx-ax)*t,ay+(by-ay)*t
                            line([p(x-.07,y,bottom+.23),p(x+.07,y,bottom+.23)],(255,179,58),2)
                    elif ax < 0 and bx < 0:
                        # Rear doors: center seam, two locking rods, hinges.
                        line([p(ax,0,bottom+.16),p(ax,0,top-.16)],(18,31,36))
                        for y in (-width*.23,width*.23):
                            line([p(ax,y,bottom+.26),p(ax,y,top-.26)],(199,211,211))
                            for z in (bottom+.5,top-.5):
                                line([p(ax,y-.11,z),p(ax,y+.11,z)],(223,225,213))
            if detailing:
                for j in range(1,max(2,round(length/.9))):
                    x=-length/2+j*length/max(2,round(length/.9))
                    line([p(x,-width*.43,top+.01),p(x,width*.43,top+.01)],tuple(min(255,v+14) for v in color))
                for y in (-width*.46,width*.46):
                    line([p(-length/2,y,top+.01),p(length/2-nose_chamfer,y,top+.01)],(149,175,175))

        def wheel(cx,cy,heading):
            # Twelve-sided upright tire and recessed hub on its outward face.
            c,s=math.cos(heading),math.sin(heading)
            for side in (-1,1):
                ny=side*.19
                if (-s*.10-c*.30)*side <= 0:
                    continue
                def ring(radius):
                    return [point(cx,cy,heading,radius*math.cos(a),ny,.48+radius*math.sin(a))
                            for a in [j*math.tau/12 for j in range(12)]]
                polygon(ring(.47),(13,17,22),(48,53,58))
                polygon(ring(.25),(122,136,144),(190,199,198))
                polygon(ring(.10),(38,49,57))
            box(cx,cy,heading,.8,.36,.15,.75,(24,28,32))

        x0,y0,h0=state.x0,state.y0,state.theta0
        x1,y1,h1=kinematics.get_trailer1_axle(state)
        x2,y2,h2=kinematics.get_trailer2_axle(state)

        def local_box(x,y,h,offset,length,width,bottom,top,color,detailing=False,nose_chamfer=0.0):
            cx,cy,_=point(x,y,h,offset,0,0)
            box(cx,cy,h,length,width,bottom,top,color,detailing,nose_chamfer)

        # Low chassis and exposed fifth-wheel decks remain visible at the joints.
        local_box(x0,y0,h0,1.4,6.0,dims.cab_width*.65,.4,.82,(49,58,66))
        local_box(x1,y1,h1,2.65,7.7,dims.trailer1_width*.67,.52,.91,(44,56,64))
        local_box(x2,y2,h2,2.4,8.1,dims.trailer2_width*.67,.52,.91,(44,56,64))
        for x,y,h,width,offsets in [
            (x0,y0,h0,dims.cab_width,(-1.05,0)),
            (x1,y1,h1,dims.trailer1_width,(-.58,.58)),
            (x2,y2,h2,dims.trailer2_width,(-.58,.58)),
        ]:
            for offset in offsets:
                for side in (-width*.47,width*.47):
                    wx,wy,_=point(x,y,h,offset,side,0)
                    wheel(wx,wy,h)
        for side in (-dims.cab_width*.5,dims.cab_width*.5):
            wx,wy,_=point(x0,y0,h0,dims.L0,side,0)
            wheel(wx,wy,h0+state.phi)

        # Short lead box, exposed coupling deck, then the longer rear box.
        local_box(x0,y0,h1,self.LEAD_CENTER,self.LEAD_LENGTH,dims.trailer1_width,1.02,3.5,(73,126,164),True,self.LEAD_NOSE_CHAMFER)
        local_box(x2,y2,h2,dims.L2+.85-7.7/2,7.7,dims.trailer2_width,1.02,3.5,(72,143,125),True)
        for x,y,h in ((x0,y0,h0),(x1,y1,h1)):
            local_box(x,y,h,0,.85,.9,.9,1.05,(132,144,149))

        # Sleeper, sloped windscreen and lower bonnet give the cab a silhouette.
        local_box(x0,y0,h0,1.65,1.95,dims.cab_width,.85,2.95,(166,40,42))
        local_box(x0,y0,h0,1.45,1.25,dims.cab_width*.76,2.95,3.12,(195,56,53))
        def cp(x,y,z):
            return point(x0,y0,h0,x,y,z)
        polygon([cp(2.62,-1.04,2.85),cp(2.62,1.04,2.85),cp(2.95,1.0,1.65),cp(2.95,-1.0,1.65)],(34,64,79),(146,176,183))
        line([cp(2.64,0,2.8),cp(2.93,0,1.7)],(14,27,35))
        # Side windows and door handles are attached to the cab walls.
        for side in (-1,1):
            normal=(-math.sin(h0)*side,math.cos(h0)*side)
            if normal[0]*.10-normal[1]*.30 > 0:
                y=side*dims.cab_width/2
                polygon([cp(1.7,y,2.7),cp(2.55,y,2.7),cp(2.55,y,1.85),cp(1.7,y,1.85)],(28,52,65),(168,181,173))
                line([cp(1.86,y,1.62),cp(2.15,y,1.62)],(213,222,215))
            # Steps and cylindrical-looking silver fuel tanks below the doors.
            tx,ty,_=cp(.9,side*1.05,.3)
            box(tx,ty,h0,1.45,.35,.35,.85,(153,166,169))
            mx,my,_=cp(2.6,side*1.45,2.0)
            box(mx,my,h0,.38,.18,1.9,2.45,(67,80,86))
            line([cp(2.55,side*1.1,2.2),cp(2.6,side*1.45,2.2)],(172,181,169))
        local_box(x0,y0,h0,3.65,1.7,dims.cab_width*.76,.8,1.65,(155,34,37))
        line([cp(2.85,0,1.66),cp(4.45,0,1.66)],(219,87,77))
        local_box(x0,y0,h0,4.5,.16,dims.cab_width*.94,.55,.86,(184,196,196))
        polygon([cp(4.51,-.57,.9),cp(4.51,.57,.9),cp(4.51,.57,1.55),cp(4.51,-.57,1.55)],(31,42,50),(185,198,194))
        for z in (1.03,1.2,1.37):
            line([cp(4.52,-.5,z),cp(4.52,.5,z)],(124,143,149))
        for side in (-.78,.78):
            polygon([cp(4.53,side-.13,.96),cp(4.53,side+.13,.96),cp(4.53,side+.13,1.25),cp(4.53,side-.13,1.25)],(255,242,204),(77,88,87))
        # Exhaust stack, behind the sleeper, capped with a dark opening.
        ex,ey,_=cp(.5,-.94,0)
        box(ex,ey,h0,.18,.18,.85,3.35,(157,170,176))
        box(ex,ey,h0,.22,.22,3.35,3.4,(30,39,44))

        # Rear underrun bar and inset lamp clusters at their actual ground emitters.
        rear_offset=dims.L2+.85-7.7-.08
        local_box(x2,y2,h2,rear_offset,.18,dims.trailer2_width,.42,.85,(78,91,96))
        for side,color in [(-1.08,(242,49,49)),(-.67,(238,232,206)),(.67,(238,232,206)),(1.08,(242,49,49))]:
            def rp(y,z):
                return point(x2,y2,h2,rear_offset-.1,y,z)
            polygon([rp(side-.12,.53),rp(side+.12,.53),rp(side+.12,.77),rp(side-.12,.77)],color,(28,35,39))

        for _,points,color,outline,width in sorted(faces,key=lambda item:round(item[0],9)):
            pixels=[project(p) for p in points]
            if width:
                pygame.draw.lines(surface,color,False,pixels,width)
            else:
                pygame.draw.polygon(surface,color,pixels)
                if outline:
                    pygame.draw.polygon(surface,outline,pixels,1)
