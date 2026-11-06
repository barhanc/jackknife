"""Reserved screen chrome for the simulation; the yard never sits underneath it."""
import math

import numpy as np
import pygame


PANEL = (21, 22, 23)
BORDER = (72, 73, 72)
TEXT = (233, 233, 226)
MUTED = (161, 164, 161)
AMBER = (255, 190, 73)
KEYCAP = (12, 13, 14)
GREEN = (135, 225, 130)
RED = (255, 85, 112)


class GameHUD:
    def __init__(self, game):
        self.game = game
        self.diagnostics_scroll = 0
        self.scroll_limit = 0
        self.alert_rect = pygame.Rect(0, 0, 0, 0)
        self.alert_reset_rect = pygame.Rect(0, 0, 0, 0)
        self.histogram_plot_rect = pygame.Rect(0, 0, 0, 0)
        self.expected_steering_x = None
        self.rays_button_rect = pygame.Rect(0, 0, 0, 0)
        self.layout()

    def layout(self):
        g = self.game
        s = g.ui_scale
        self.pad, self.gap = max(8, int(16*s)), max(6, int(8*s))
        self.row_h = int(40*s)
        self.button_rects = {}
        specs = [
            ("mode", "MANUAL", "M"),
            ("draw", "DRAW", "E"), ("sample", "NEW ROUTE", "P"),
            ("speed", "16X", "T"), ("pause", "CONTINUE", "Space"),
            ("diagnostics", "TELEMETRY", "H"),
        ]
        brand_w = g.font_arcade_title.size("JACKKNIFE")[0] + int(24*s)
        widths = [brand_w]
        for name, text, key in specs:
            widths.append(g.font_arcade.size(text)[0] + g.font_sm.size(key)[0] + 4*self.gap)
        cells, bottom = self._flow(widths, self.pad, self.pad, g.width-self.pad, self.row_h)
        if cells[-1].y == cells[0].y:
            extra = g.width-self.pad-cells[-1].right
            cells = [cells[0]] + [r.move(extra, 0) for r in cells[1:]]
        self.brand_rect = cells[0]
        for (name, _, _), rect in zip(specs, cells[1:]):
            self.button_rects[name] = rect
        self.toolbar_rect = pygame.Rect(0, 0, g.width, bottom+self.pad)
        self.cancel_rect = pygame.Rect(g.width-self.pad-int(96*s), self.pad, int(96*s), self.row_h)

        self.status_widths = [
            g.font_arcade.size("AUTONOMOUS")[0] + int(20*s),
            g.font_arcade_small.size("STABILITY")[0] + int(128*s),
            g.font_arcade_small.size("INFERENCE")[0] + g.font_mono.size("999.9 ms")[0] + 2*self.gap,
            g.font_arcade.size("RESET")[0] + g.font_sm.size("R")[0] + 4*self.gap,
        ]
        self.status_cells, bottom = self._flow(self.status_widths, self.pad, self.gap, g.width-self.pad, int(32*s))
        status_h = bottom+self.gap
        self.status_rect = pygame.Rect(0, g.height-status_h, g.width, status_h)
        self.status_cells = [r.move(0, self.status_rect.y) for r in self.status_cells]
        if self.status_cells[-1].y == self.status_cells[0].y:
            extra = g.width-2*self.pad-sum(self.status_widths)-3*self.gap
            self.status_cells = [r.move(round(i*extra/3), 0) for i, r in enumerate(self.status_cells)]
        self.reset_rect = self.status_cells[-1]
        content = pygame.Rect(0, self.toolbar_rect.bottom, g.width, max(1, self.status_rect.top-self.toolbar_rect.bottom))
        self.drawer_rect = pygame.Rect(0, 0, 0, 0)
        self.yard_rect = content.copy()
        self.drawer_horizontal = g.width < int(900*s)
        if g.show_help:
            if self.drawer_horizontal:
                height = min(int(250*s), content.h//2)
                self.drawer_rect = pygame.Rect(0, content.bottom-height, content.w, height)
                self.yard_rect.height -= height
            else:
                width = int(300*s)
                self.drawer_rect = pygame.Rect(content.right-width, content.top, width, content.h)
                self.yard_rect.width -= width

    def _flow(self, widths, left, top, right, height):
        x, y = left, top
        cells = []
        for width in widths:
            if x > left and x+width > right:
                x, y = left, y+height+self.gap
            cells.append(pygame.Rect(x, y, min(width, max(1, right-left)), height))
            x += width+self.gap
        return cells, y+height

    @staticmethod
    def _glyph(text, font, color):
        surface = font.render(text, False, color)
        ink = surface.get_bounding_rect()
        return surface.subsurface(ink) if ink.w and ink.h else surface

    def _text(self, text, rect, font=None, color=TEXT, center=False):
        font = font or self.game.font_mono
        surface = self._glyph(text, font, color)
        while text and surface.get_width() > rect.w:
            text = text[:-1]
            surface = self._glyph(text, font, color)
        target = surface.get_rect(center=rect.center) if center else surface.get_rect(midleft=rect.midleft)
        self.game.screen.blit(surface, target)

    @staticmethod
    def _corners(rect, cut=4):
        x, y, w, h = rect
        return [(x+cut,y), (x+w-cut-1,y), (x+w-1,y+cut), (x+w-1,y+h-cut-1), (x+w-cut-1,y+h-1), (x+cut,y+h-1), (x,y+h-cut-1), (x,y+cut)]

    def _panel(self, rect):
        screen = self.game.screen
        pygame.draw.rect(screen, PANEL, rect)
        for y in range(rect.top+2, rect.bottom, max(4, int(4*self.game.ui_scale))):
            pygame.draw.line(screen, (25, 26, 27), (rect.left, y), (rect.right-1, y))

    def _button(self, rect, label, key="", active=False, primary=False):
        screen = self.game.screen
        hover = rect.collidepoint(pygame.mouse.get_pos())
        fill = (49, 43, 31) if primary else ((47, 49, 49) if hover or active else (29, 30, 31))
        pygame.draw.polygon(screen, fill, self._corners(rect))
        pygame.draw.polygon(screen, AMBER if primary else AMBER if hover or active else BORDER, self._corners(rect), 1)
        pygame.draw.line(screen, (104, 83, 45) if primary else (58, 60, 60), (rect.left+5,rect.bottom-3), (rect.right-6,rect.bottom-3))
        if key:
            key_w = self.game.font_sm.size(key)[0] + self.gap
            label_w = self._glyph(label, self.game.font_arcade, TEXT).get_width()
            group_w = label_w+self.gap+key_w
            group_left = rect.centerx-group_w//2
            key_rect = pygame.Rect(group_left+label_w+self.gap, rect.centery-int(10*self.game.ui_scale), key_w, int(20*self.game.ui_scale))
            pygame.draw.rect(screen, KEYCAP, key_rect)
            self._text(key, key_rect, self.game.font_sm, MUTED, True)
            text_rect = pygame.Rect(group_left, rect.top, label_w, rect.h)
        else:
            text_rect = rect.inflate(-2*self.gap, 0)
        self._text(label, text_rect, self.game.font_arcade, color=AMBER if primary else TEXT, center=True)

    def click(self, pos):
        """Return True when screen chrome consumes a click."""
        g = self.game
        if not g.drawing_mode and self.alert_rect.collidepoint(pos) and self._alert_state() is not None:
            if self.alert_reset_rect.collidepoint(pos):
                if g.has_arrived or g.is_off_track or g.kinematics.is_jackknifed(g.state):
                    g.reset_vehicle_to_path()
                g.is_paused = False
            return True
        if self.reset_rect.collidepoint(pos):
            if g.drawing_mode:
                g.finish_drawing(cancel=True)
            g.reset_vehicle_to_path()
            return True
        if self.toolbar_rect.collidepoint(pos):
            if g.drawing_mode:
                if self.cancel_rect.collidepoint(pos):
                    g.finish_drawing(cancel=True)
                return True
            for name, rect in self.button_rects.items():
                if not rect.collidepoint(pos):
                    continue
                if name == "mode":
                    g.toggle_control_mode()
                elif name == "draw":
                    g.start_drawing_mode()
                elif name == "sample":
                    g.sample_random_prior_path()
                elif name == "speed":
                    g.cycle_speed()
                elif name == "pause":
                    g.is_paused = not g.is_paused
                elif name == "diagnostics":
                    g.show_help = not g.show_help
                    g.update_hud_layout()
                break
            return True
        if self.drawer_rect.collidepoint(pos):
            if self.rays_button_rect.collidepoint(pos):
                g.show_rays = not g.show_rays
            return True
        return self.status_rect.collidepoint(pos)

    def scroll(self, wheel_y):
        self.diagnostics_scroll = max(0, min(self.scroll_limit, self.diagnostics_scroll-int(wheel_y*36*self.game.ui_scale)))

    def draw(self):
        g = self.game
        self._panel(self.toolbar_rect)
        pygame.draw.line(g.screen, BORDER, self.toolbar_rect.bottomleft, self.toolbar_rect.bottomright, 2)
        if g.drawing_mode:
            self._text("DRAW ROUTE", pygame.Rect(self.pad, self.pad, max(1, self.cancel_rect.left-2*self.pad), self.row_h), g.font_arcade, AMBER)
            hint = "Drag · Release to finish · Esc cancel"
            self._text(hint, pygame.Rect(self.pad, self.toolbar_rect.bottom-int(25*g.ui_scale), g.width-2*self.pad, int(20*g.ui_scale)), g.font_sm, MUTED)
            self._button(self.cancel_rect, "CANCEL", primary=True)
        else:
            title = self.brand_rect.copy(); title.height = int(25*g.ui_scale)
            self._text("JACKKNIFE", title.move(1, 2), g.font_arcade_title, (87, 67, 35))
            self._text("JACKKNIFE", title, g.font_arcade_title, AMBER)
            self._text("B-DOUBLE // REVERSE", pygame.Rect(title.x, title.bottom, self.brand_rect.w, int(15*g.ui_scale)), g.font_sm, MUTED)
            for name, label, key in [
                ("mode", "AUTO" if g.mode == "AUTONOMOUS" else "MANUAL", "M"),
                ("draw", "DRAW", "E"), ("sample", "NEW ROUTE", "P"),
                ("speed", f"{g.time_multiplier}X", "T"),
                ("pause", "CONTINUE" if g.is_paused else "PAUSE", "Space"),
                ("diagnostics", "TELEMETRY", "H"),
            ]:
                active = (name == "diagnostics" and g.show_help) or (name == "mode" and g.mode == "AUTONOMOUS")
                self._button(self.button_rects[name], label, key, active=active, primary=name == "draw")
        self._draw_status()
        if self.drawer_rect.w:
            self._draw_diagnostics()
        self._draw_alert()

    def _draw_status(self):
        g = self.game
        self._panel(self.status_rect)
        pygame.draw.line(g.screen, BORDER, self.status_rect.topleft, self.status_rect.topright, 2)
        color = AMBER
        if g.drawing_mode:
            status = "DRAW ROUTE"
        elif g.kinematics.is_jackknifed(g.state):
            status, color = "JACKKNIFED", RED
        elif g.has_arrived:
            status, color = "COMPLETE", GREEN
        elif g.is_paused:
            status, color = "PAUSED", AMBER
        else:
            status = "AUTONOMOUS" if g.mode == "AUTONOMOUS" else "MANUAL"
        self._text(status, self.status_cells[0], g.font_arcade, color)
        rect = self.status_cells[1]
        label_w = g.font_arcade_small.size("STABILITY")[0]
        self._text("STABILITY", pygame.Rect(rect.x, rect.y, label_w, rect.h), g.font_arcade_small, MUTED)
        rho = g.kinematics.jackknife_proximity(g.state)
        bar = pygame.Rect(rect.x+label_w+self.gap, rect.centery-int(4*g.ui_scale), int(112*g.ui_scale), int(8*g.ui_scale))
        self.stability_bar_rect = bar
        for i in range(14):
            left = bar.x+round(i*bar.w/14)
            right = bar.x+round((i+1)*bar.w/14)-max(1, int(2*g.ui_scale))
            segment = pygame.Rect(left, bar.y, max(1, right-left), bar.h)
            color = (GREEN if rho > .45 else AMBER if rho > .2 else RED) if i/14 < rho else (53, 56, 56)
            pygame.draw.rect(g.screen, color, segment)
        rect = self.status_cells[2]
        label_w = g.font_arcade_small.size("INFERENCE")[0]
        self._text("INFERENCE", pygame.Rect(rect.x, rect.y, label_w, rect.h), g.font_arcade_small, MUTED)
        self._text(f"{g.inference_time_ms:.1f} ms", pygame.Rect(rect.x+label_w+self.gap, rect.y, rect.w-label_w-self.gap, rect.h), g.font_mono, AMBER)
        self._button(self.reset_rect, "RESET", "R")

    def _alert_state(self):
        g = self.game
        if g.drawing_mode:
            return None
        if g.kinematics.is_jackknifed(g.state):
            return "JACKKNIFED", "HITCH LIMIT REACHED", RED
        if g.is_off_track:
            return "OFF TRACK", "RETRY OR DRAW A WIDER ROUTE", RED
        if g.has_arrived:
            return "DOCK SECURED", "DELIVERY COMPLETE", GREEN
        if g.is_paused:
            return "PAUSED", "SPACE TO CONTINUE", AMBER
        return None

    def _draw_alert(self):
        g = self.game
        alert = self._alert_state()
        self.alert_rect = pygame.Rect(0, 0, 0, 0)
        self.alert_reset_rect = pygame.Rect(0, 0, 0, 0)
        if alert is None:
            return
        title, subtitle, accent = alert
        view = self.yard_rect.copy()
        if view.h < int(170*g.ui_scale) or view.w < int(280*g.ui_scale):
            view = pygame.Rect(0, self.toolbar_rect.bottom, g.width, self.status_rect.top-self.toolbar_rect.bottom)
        width = min(int(520*g.ui_scale), view.w-2*self.pad)
        height = min(int(172*g.ui_scale), view.h-2*self.gap)
        rect = pygame.Rect(0, 0, max(1,width), max(1,height))
        rect.center = view.center
        self.alert_rect = rect
        pygame.draw.polygon(g.screen, (8, 9, 10), self._corners(rect.move(4, 5), 8))
        pygame.draw.polygon(g.screen, PANEL, self._corners(rect, 8))
        pygame.draw.polygon(g.screen, accent, self._corners(rect, 8), 2)
        font = g.font_arcade_alert
        if font.size(title)[0] > rect.w-2*self.pad:
            font = g.font_arcade_title
        if font.size(title)[0] > rect.w-2*self.pad:
            font = g.font_arcade
        compact = rect.h < int(156*g.ui_scale)
        title_rect = pygame.Rect(rect.x+self.pad, rect.y+int((12 if compact else 24)*g.ui_scale), rect.w-2*self.pad, int(34*g.ui_scale))
        self._text(title, title_rect.move(2, 2), font, (65, 53, 32), True)
        self._text(title, title_rect, font, accent, True)
        if not compact:
            self._text(subtitle, pygame.Rect(rect.x+self.pad, title_rect.bottom+int(8*g.ui_scale), rect.w-2*self.pad, int(22*g.ui_scale)), g.font_mono, TEXT, True)
        self.alert_reset_rect = pygame.Rect(0, 0, min(int(180*g.ui_scale),rect.w-2*self.pad), int(34*g.ui_scale))
        self.alert_reset_rect.midbottom = rect.centerx, rect.bottom-int(16*g.ui_scale)
        retry = g.has_arrived or g.is_off_track or g.kinematics.is_jackknifed(g.state)
        self._button(self.alert_reset_rect, "RETRY" if retry else "CONTINUE", "R" if retry else "Space", primary=True)

    @staticmethod
    def steering_bin_x(angle, plot):
        # Steering classes are bin centers, not histogram edges.
        return plot.x+plot.w/30 + (angle+35)/70*(plot.w*14/15)

    def _steering_axis(self, plot, histogram):
        g = self.game
        self.axis_ticks = []
        for angle in (-35, 0, 35):
            x = self.steering_bin_x(angle, plot) if histogram else plot.left+(angle+35)/70*plot.w
            x = round(x)
            self.axis_ticks.append((angle, x))
            pygame.draw.line(g.screen, MUTED, (x,plot.bottom+2), (x,plot.bottom+6))
            surface = self._glyph(f"{angle:+d}°" if angle else "0°", g.font_sm, MUTED)
            g.screen.blit(surface, surface.get_rect(midtop=(x,plot.bottom+int(10*g.ui_scale))))

    def _draw_diagnostics(self):
        g = self.game
        area = self.drawer_rect
        self._panel(area)
        pygame.draw.rect(g.screen, BORDER, area, 1)
        old_clip = g.screen.get_clip()
        g.screen.set_clip(area)
        inner = area.inflate(-2*self.pad, -2*self.pad)
        heading = pygame.Rect(inner.x, inner.y, inner.w, int(24*g.ui_scale))
        body = pygame.Rect(inner.x, inner.y+int(36*g.ui_scale), inner.w, max(1, inner.h-int(36*g.ui_scale)))
        horizontal = self.drawer_horizontal and g.width >= 600
        section_w = inner.w//2-self.gap if horizontal else inner.w
        description = "Lines show which path points the AI is looking at. Brighter lines mean more attention."
        lines = []
        line = ""
        for word in description.split():
            candidate = f"{line} {word}".strip()
            if line and g.font_sm.size(candidate)[0] > section_w:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        chart_h = int(128*g.ui_scale)
        rays_h = int(44*g.ui_scale)+len(lines)*int(16*g.ui_scale)
        controls_h = int(256*g.ui_scale)
        left_h = chart_h+int(20*g.ui_scale)+rays_h
        required_h = max(body.h, max(left_h,controls_h) if horizontal else left_h+int(24*g.ui_scale)+controls_h)
        self.scroll_limit = max(0, required_h-body.h)
        self.diagnostics_scroll = min(self.diagnostics_scroll, self.scroll_limit)
        self._text("DIAGNOSTICS · H CLOSE" + (" · SCROLL" if self.scroll_limit else ""), heading, g.font_badge, MUTED)
        g.screen.set_clip(body.clip(area))
        top = body.y-self.diagnostics_scroll
        chart = pygame.Rect(inner.x,top,section_w,chart_h)
        rays = pygame.Rect(chart.x,chart.bottom+int(20*g.ui_scale),section_w,rays_h)
        if horizontal:
            controls = pygame.Rect(chart.right+2*self.gap,top,inner.right-chart.right-2*self.gap,controls_h)
        else:
            controls = pygame.Rect(inner.x,rays.bottom+int(24*g.ui_scale),inner.w,controls_h)
        self._text("STEERING", pygame.Rect(chart.x, chart.y, chart.w, int(22*g.ui_scale)), g.font_arcade, AMBER)
        angle = math.degrees(g.state.phi)
        label_pad = g.font_sm.size("-35°")[0]//2+2
        plot = pygame.Rect(chart.x+label_pad, chart.y+int(34*g.ui_scale), max(1,chart.w-2*label_pad), int(64*g.ui_scale))
        self.expected_steering_x = None
        self.manual_steering_bar_rect = pygame.Rect(0, 0, 0, 0)
        if g.mode == "AUTONOMOUS" or g.ai_save_active:
            angle = float(np.dot(g.action_probs, np.linspace(-35, 35, 15)))
            self.histogram_plot_rect = plot.copy()
            maximum = max(.2, float(np.max(g.action_probs)))
            for i, probability in enumerate(g.action_probs):
                x0 = plot.x+round(i*plot.w/15)
                x1 = plot.x+round((i+1)*plot.w/15)-2
                height = max(2, round(probability/maximum*plot.h))
                pygame.draw.rect(g.screen, AMBER if i == int(np.argmax(g.action_probs)) else (72, 77, 77), (x0, plot.bottom-height, max(1, x1-x0), height))
            needle_x = round(self.steering_bin_x(angle, plot))
            self.expected_steering_x = needle_x
            line_width = max(1, 2*round(g.ui_scale)-1)
            radius = max(3, round(4*g.ui_scale))
            pygame.draw.line(g.screen, GREEN, (needle_x,plot.top-3), (needle_x,plot.bottom+2), line_width)
            pygame.draw.polygon(g.screen, GREEN, [(needle_x-radius,plot.top-8),(needle_x+radius,plot.top-8),(needle_x,plot.top-3)])
            self._steering_axis(plot, histogram=True)
        else:
            self.histogram_plot_rect = pygame.Rect(0, 0, 0, 0)
            bar = pygame.Rect(plot.x, plot.centery-int(6*g.ui_scale), plot.w, int(12*g.ui_scale))
            self.manual_steering_bar_rect = bar
            pygame.draw.rect(g.screen, (53, 56, 56), bar)
            cursor = round(plot.left+(max(-35,min(35,angle))+35)/70*plot.w)
            self.manual_steering_cursor_x = cursor
            middle = plot.centerx
            pygame.draw.rect(g.screen, AMBER, (min(middle,cursor),bar.y,max(1,abs(cursor-middle)),bar.h))
            pygame.draw.line(g.screen, TEXT, (middle,bar.y-4),(middle,bar.bottom+4))
            pygame.draw.line(g.screen, AMBER, (cursor,bar.y-5),(cursor,bar.bottom+5),2)
            self._steering_axis(plot, histogram=False)
        toggle_w = g.font_arcade.size("OFF")[0]+g.font_sm.size("V")[0]+4*self.gap
        toggle = pygame.Rect(rays.right-toggle_w,rays.top,toggle_w,int(30*g.ui_scale))
        self._text("ATTENTION RAYS",pygame.Rect(rays.x,rays.y,rays.w-toggle_w-self.gap,toggle.h),g.font_arcade,AMBER)
        self._button(toggle,"ON" if g.show_rays else "OFF","V",active=g.show_rays)
        self.rays_button_rect = toggle.clip(body).clip(area)
        for i,line in enumerate(lines):
            row = pygame.Rect(rays.x,rays.y+int((44+16*i)*g.ui_scale),rays.w,int(16*g.ui_scale))
            self._text(line,row,g.font_sm,MUTED)
        for section in (rays,controls):
            if section.y > chart.y:
                pygame.draw.line(g.screen,BORDER,(section.left,section.top-int(10*g.ui_scale)),(section.right,section.top-int(10*g.ui_scale)))
        self._text("CONTROLS", pygame.Rect(controls.x, controls.y, controls.w, int(24*g.ui_scale)), g.font_arcade, AMBER)
        controls_list = [("W/S", "Gas / reverse"), ("A/D", "Steer"), ("M", "Auto / manual"), ("T", "1x / 2x / 4x / 8x / 16x"), ("Space", "Pause"), ("R", "Reset"), ("E", "Draw path"), ("V", "Attention rays"), ("Esc", "Cancel drawing")]
        row_h = max(14, min(int(25*g.ui_scale), (controls.h-int(32*g.ui_scale))//len(controls_list)))
        for i, (key, label) in enumerate(controls_list):
            row = pygame.Rect(controls.x, controls.y+int(32*g.ui_scale)+i*row_h, controls.w, row_h)
            key_w = min(int(55*g.ui_scale), row.w//3)
            self._text(key, pygame.Rect(row.x, row.y, key_w, row.h), g.font_sm, AMBER)
            self._text(label, pygame.Rect(row.x+key_w, row.y, max(1, row.w-key_w), row.h), g.font_sm, MUTED)
        g.screen.set_clip(old_clip)
