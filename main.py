import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

import numpy as np
from pyrr import Matrix44
import math

import tkinter as tk
from tkinter import filedialog

from mesh import (
    load_mesh,
    MeshInfo
)

# --- SHADER SOURCE ---
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 mvp;
out vec3 vColor;

void main() {
    gl_Position = mvp * vec4(aPos, 1.0);
    vColor = aColor;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;

uniform vec3 overrideColor;
uniform bool useOverride;

void main() {
    if (useOverride) {
        FragColor = vec4(overrideColor, 1.0);
    } else {
        FragColor = vec4(vColor, 1.0);
    }
}
"""

class MeshViewer:
    def __init__(self):
        self.mode = 0
        self.mesh = None
        self.intersected_face_ids = None

        self.last_o_state = glfw.RELEASE
        self.last_j_state = glfw.RELEASE
        self.last_k_state = glfw.RELEASE
        self.last_l_state = glfw.RELEASE

        if not glfw.init():
            raise Exception("GLFW could not be initialized!")

        self.window = glfw.create_window(1000, 800, "Mesh Viewer | O: Open File", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window could not be created!")

        glfw.make_context_current(self.window)
        glEnable(GL_DEPTH_TEST)
        
        self.shader = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )

        # Buffer Set 1: Main
        self.main_vao = glGenVertexArrays(1)
        self.main_vbo = glGenBuffers(1)
        self.main_ebo = glGenBuffers(1)
        self.main_index_count = 0

        # Buffer Set 2: Intersected Faces
        self.intersected_vao = glGenVertexArrays(1)
        self.intersected_vbo = glGenBuffers(1)
        self.intersected_ebo = glGenBuffers(1)
        self.intersected_index_count = 0
        
        self.index_count = 0

        self.mvp_loc = glGetUniformLocation(self.shader, "mvp")
        self.override_loc = glGetUniformLocation(self.shader, "overrideColor")
        self.use_override_loc = glGetUniformLocation(self.shader, "useOverride")

    def open_file_dialog(self):
        # Create a hidden Tkinter root window
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Select Mesh File",
            filetypes=[("Mesh Files", "*.obj *.stl *.ply *.glb *.off"), ("All Files", "*.*")]
        )
        
        root.destroy()
        
        if file_path:
            self.load_mesh(file_path)

    def load_mesh(self, path):
        try:
            mesh = load_mesh(path)
            self.mesh = mesh
            self.mesh_info = MeshInfo(mesh)
            self.intersected_face_ids = self.mesh_info.intersected_face_ids
            
            self.update_gpu_buffers()

            print(f"Loaded mesh: {path}")
            print(self.mesh_info)
            
        except Exception as e:
            print(f"Failed to load mesh: {e}")
    
    def update_gpu_buffers(self):
        if self.mesh is None: return

        # Split faces into two groups
        all_indices = np.arange(len(self.mesh.faces))
        intersected_mask = np.array([i in self.intersected_face_ids for i in all_indices])
        
        # 1. Prepare Main Mesh (Not highlighted)
        main_faces = self.mesh.faces[~intersected_mask]
        self.main_index_count = self.setup_buffer(self.main_vao, self.main_vbo, self.main_ebo, main_faces)

        # 2. Prepare Intersected Faces (Selected)
        intersected_faces = self.mesh.faces[intersected_mask]
        self.intersected_index_count = self.setup_buffer(self.intersected_vao, self.intersected_vbo, self.intersected_ebo, intersected_faces)

    def setup_buffer(self, vao, vbo, ebo, faces):
        if len(faces) == 0: return 0
        
        # Un-index vertices so each triangle has unique data (easier for coloring/normals)
        vertices = self.mesh.vertices[faces].reshape(-1, 3)
        colors = np.full((vertices.shape[0], 3), 0.7, dtype=np.float32)
        
        data = np.hstack((vertices, colors)).astype(np.float32)
        indices = np.arange(len(vertices)).astype(np.uint32)

        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_DYNAMIC_DRAW)

        # Layout: Pos(3), Color(3)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        
        return len(indices)

    def handle_input(self):
        # Handle 'J' for Mode 0 (SOLID)
        j_state = glfw.get_key(self.window, glfw.KEY_J)
        if j_state == glfw.PRESS and self.last_j_state == glfw.RELEASE:
            self.mode = 0
        self.last_j_state = j_state

        # Handle 'K' for Mode 1 (WIREFRAME)
        k_state = glfw.get_key(self.window, glfw.KEY_K)
        if k_state == glfw.PRESS and self.last_k_state == glfw.RELEASE:
            self.mode = 1
        self.last_k_state = k_state

        # Handle 'L' for Mode 2 (BOTH)
        l_state = glfw.get_key(self.window, glfw.KEY_L)
        if l_state == glfw.PRESS and self.last_l_state == glfw.RELEASE:
            self.mode = 2
        self.last_l_state = l_state

        # Handle 'O' for Open
        o_state = glfw.get_key(self.window, glfw.KEY_O)
        if o_state == glfw.PRESS and self.last_o_state == glfw.RELEASE:
            self.open_file_dialog()
        self.last_o_state = o_state
    
    def render_mesh(self):
        glUseProgram(self.shader)

        proj = Matrix44.perspective_projection(45.0, 1000/800, 0.1, 100.0)
        cam_x = math.sin(glfw.get_time() * 0.3) * 3.5
        cam_z = math.cos(glfw.get_time() * 0.3) * 3.5
        view = Matrix44.look_at([cam_x, 1.5, cam_z], [0, 0, 0], [0, 1, 0])
        model = Matrix44.identity() # Object is already centered and scaled
        
        mvp = proj * view * model
        glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, mvp)

        # --- DRAW PASS 1: Main Mesh ---
        if self.main_index_count > 0:
            glBindVertexArray(self.main_vao)
            glUniform1i(self.use_override_loc, False)
            
            if self.mode == 0 or self.mode == 2: # Solid
                # Add offset to push solid faces away from lines/highlighter
                glEnable(GL_POLYGON_OFFSET_FILL)
                glPolygonOffset(1.0, 1.0)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glDrawElements(GL_TRIANGLES, self.main_index_count, GL_UNSIGNED_INT, None)
                glDisable(GL_POLYGON_OFFSET_FILL)
            
            if self.mode == 1 or self.mode == 2: # Wireframe
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glUniform1i(self.use_override_loc, True)
                glUniform3f(self.override_loc, 0.5, 0.5, 0.5)
                glDrawElements(GL_TRIANGLES, self.main_index_count, GL_UNSIGNED_INT, None)

        # --- DRAW PASS 2: Highlighted Part ---
        if self.intersected_index_count > 0:
            glBindVertexArray(self.intersected_vao)
                
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glUniform1i(self.use_override_loc, True)
            glUniform3f(self.override_loc, 1.0, 0.5, 0.0) # ORANGE
            
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(1.0, 1.0)
            glDrawElements(GL_TRIANGLES, self.intersected_index_count, GL_UNSIGNED_INT, None)
            glDisable(GL_POLYGON_OFFSET_FILL)

            # 2. Wireframe Outline for Highlight
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glUniform1i(self.use_override_loc, True) # Keep override on for white color
            glUniform3f(self.override_loc, 1.0, 1.0, 1.0) # White outline for selected
            glDrawElements(GL_TRIANGLES, self.intersected_index_count, GL_UNSIGNED_INT, None)

    def run(self):
        while not glfw.window_should_close(self.window):
            self.handle_input()
            
            glClearColor(0.05, 0.05, 0.1, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            if self.mesh is not None:
                self.render_mesh()

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

if __name__ == "__main__":
    app = MeshViewer()
    app.run()