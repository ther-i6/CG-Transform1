import taichi as ti
import math

ti.init(arch=ti.gpu)

# 窗口设置
width, height = 700, 700
gui = ti.GUI("CG-MVP Transform", res=(width, height))

# 三角形顶点
vertices = ti.Vector.field(4, dtype=ti.f32, shape=3)
screen_pos = ti.Vector.field(2, dtype=ti.f32, shape=3)


@ti.kernel
def init_vertices():
    vertices[0] = ti.Vector([2.0, 0.0, -2.0, 1.0])
    vertices[1] = ti.Vector([0.0, 2.0, -2.0, 1.0])
    vertices[2] = ti.Vector([-2.0, 0.0, -2.0, 1.0])


init_vertices()


# ==================== MVP 矩阵 ====================
@ti.func
def get_model_matrix(angle):
    rad = angle * math.pi / 180
    c, s = ti.cos(rad), ti.sin(rad)
    return ti.math.mat4(
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    )


@ti.func
def get_view_matrix(eye_pos):
    x, y, z = eye_pos
    return ti.math.mat4(
        [1, 0, 0, -x],
        [0, 1, 0, -y],
        [0, 0, 1, -z],
        [0, 0, 0, 1]
    )


@ti.func
def get_projection_matrix(fovy, aspect, zNear, zFar):
    fov_rad = fovy * math.pi / 180
    t = ti.tan(fov_rad / 2) * abs(-zNear)
    b = -t
    r = aspect * t
    l = -r
    n, f = -zNear, -zFar

    persp2ortho = ti.math.mat4(
        [n, 0, 0, 0],
        [0, n, 0, 0],
        [0, 0, n + f, -n * f],
        [0, 0, 1, 0]
    )

    ortho = ti.math.mat4(
        [2 / (r - l), 0, 0, -(r + l) / (r - l)],
        [0, 2 / (t - b), 0, -(t + b) / (t - b)],
        [0, 0, 2 / (n - f), -(n + f) / (n - f)],
        [0, 0, 0, 1]
    )
    return ortho @ persp2ortho


@ti.kernel
def update(angle: ti.f32):
    eye = ti.Vector([0, 0, 5])
    aspect = width / height
    model = get_model_matrix(angle)
    view = get_view_matrix(eye)
    proj = get_projection_matrix(45, aspect, 0.1, 50)
    mvp = proj @ view @ model

    for i in range(3):
        clip = mvp @ vertices[i]
        ndc = clip / clip.w
        x = (ndc.x + 1) * 0.5
        y = (ndc.y + 1) * 0.5
        screen_pos[i] = ti.Vector([x, y])


# ==================== 主循环 ====================
angle = 0.0
while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            break
        if gui.event.key == 'a':
            angle += 10
        if gui.event.key == 'd':
            angle -= 10

    update(angle)
    gui.clear(0x181818)

    gui.line(screen_pos[0], screen_pos[1], color=0xFF0000, radius=4)  # 红色，粗4
    gui.line(screen_pos[1], screen_pos[2], color=0x00FF00, radius=4)  # 绿色，粗4
    gui.line(screen_pos[2], screen_pos[0], color=0x0000FF, radius=4)  # 蓝色，粗4

    gui.show()