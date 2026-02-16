import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

import numpy as np
from pyrr import Matrix44
import math

import tkinter as tk
from tkinter import filedialog
from PIL import Image

import trimesh

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
        self.mesh: trimesh.Trimesh = None
        self.intersected_face_ids = None
        
        self.show_intersected = False
        self.show_face_normals = False
        self.show_vertex_normals = False
        self.show_point_cloud = False

        self.last_o_state = glfw.RELEASE
        self.last_i_state = glfw.RELEASE
        self.last_j_state = glfw.RELEASE
        self.last_k_state = glfw.RELEASE
        self.last_l_state = glfw.RELEASE
        self.last_n_state = glfw.RELEASE
        self.last_m_state = glfw.RELEASE
        self.last_p_state = glfw.RELEASE
        self.last_c_state = glfw.RELEASE

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

        # Main Buffer
        self.main_vao = glGenVertexArrays(1)
        self.main_vbo = glGenBuffers(1)
        self.main_ebo = glGenBuffers(1)
        self.main_index_count = 0

        # Intersected Faces Buffer
        self.intersected_vao = glGenVertexArrays(1)
        self.intersected_vbo = glGenBuffers(1)
        self.intersected_ebo = glGenBuffers(1)
        self.intersected_index_count = 0

        # Face Normals Buffer (lines)
        self.face_normals_vao = glGenVertexArrays(1)
        self.face_normals_vbo = glGenBuffers(1)
        self.face_normals_count = 0

        # Vertex Normals Buffer (lines)
        self.vertex_normals_vao = glGenVertexArrays(1)
        self.vertex_normals_vbo = glGenBuffers(1)
        self.vertex_normals_count = 0

        # Point Cloud Buffer (points)
        self.point_cloud_vao = glGenVertexArrays(1)
        self.point_cloud_vbo = glGenBuffers(1)
        self.point_cloud_count = 0

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
    
    def capture_screenshot(self):
        """Capture the mesh area and save it."""
        width, height = glfw.get_window_size(self.window)
        
        # Read full framebuffer
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # Convert to PIL Image and flip vertically
        image_data = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        image_data = np.flipud(image_data)
        img = Image.fromarray(image_data, 'RGB')
        
        # Autocrop to remove empty space (black background)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        
        # Open save dialog
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.asksaveasfilename(
            title="Save Screenshot",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf"), ("All Files", "*.*")]
        )
        
        root.destroy()
        
        if file_path:
            try:
                if file_path.lower().endswith('.pdf'):
                    img.save(file_path, 'PDF')
                    print(f"PDF saved: {file_path}")
                else:
                    img.save(file_path)
                    print(f"Screenshot saved: {file_path}")
            except Exception as e:
                print(f"Failed to save screenshot: {e}")

    def load_mesh(self, path):
        try:
            mesh = load_mesh(path)
            if mesh is None:
                print("Mesh is None after loading. Check if the file is valid and supported.")
                return
            elif len(mesh.faces) == 0:
                print("Loaded mesh has no faces. Please select a valid mesh file.")
                return
            elif len(mesh.vertices) == 0:
                print("Loaded mesh has no vertices. Please select a valid mesh file.")
                return
            self.mesh = mesh

            # mesh analysis and info extraction
            self.mesh_info = MeshInfo(mesh)
            
            # Store intersected face IDs for rendering
            self.intersected_face_ids = self.mesh_info.intersected_face_ids
            
            # Calculate diagonal and normal length for visualization
            bounds = self.mesh_info.analysis["bounds"]
            self.diag = np.linalg.norm(bounds[1] - bounds[0])
            self.normal_length = max(self.diag * 0.02, 0.01)

            # Sample points for point cloud visualization (if needed)
            self.points: np.ndarray = mesh.sample(8192)
            
            self.update_gpu_buffers()

            print(f"Loaded mesh: {path}")
            print(self.mesh_info)
            
        except Exception as e:
            print(f"Failed to load mesh: {e}")
    
    def update_gpu_buffers(self):
        # Split faces into two groups
        all_indices = np.arange(len(self.mesh.faces))
        intersected_mask = np.array([i in self.intersected_face_ids for i in all_indices])
        
        # 1. Prepare Main Mesh (Not highlighted)
        main_faces = self.mesh.faces[~intersected_mask]
        self.main_index_count = self.setup_buffer(self.main_vao, self.main_vbo, self.main_ebo, main_faces)

        # 2. Prepare Intersected Faces (Selected)
        intersected_faces = self.mesh.faces[intersected_mask]
        self.intersected_index_count = self.setup_buffer(self.intersected_vao, self.intersected_vbo, self.intersected_ebo, intersected_faces)

        # 3. Prepare Normals
        self.face_normals_count = self.setup_face_normals_buffer()
        self.vertex_normals_count = self.setup_vertex_normals_buffer()
        self.point_cloud_count = self.setup_point_cloud_buffer()

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

    def setup_point_cloud_buffer(self):
        points = self.points
        colors = np.full((points.shape[0], 3), 1.0, dtype=np.float32)  # White points
        data = np.hstack((points, colors)).astype(np.float32)

        glBindVertexArray(self.point_cloud_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.point_cloud_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        return points.shape[0]

    def setup_face_normals_buffer(self):
        length = self.normal_length

        # Face centers and normals
        centers = self.mesh.triangles_center
        normals = self.mesh.face_normals

        line_verts = np.empty((centers.shape[0] * 2, 3), dtype=np.float32)
        line_verts[0::2] = centers
        line_verts[1::2] = centers + normals * length

        colors = np.full((line_verts.shape[0], 3), 0.2, dtype=np.float32)
        data = np.hstack((line_verts, colors)).astype(np.float32)

        glBindVertexArray(self.face_normals_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.face_normals_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        return line_verts.shape[0]

    def setup_vertex_normals_buffer(self):
        length = self.normal_length

        verts = self.mesh.vertices
        normals = self.mesh.vertex_normals

        line_verts = np.empty((verts.shape[0] * 2, 3), dtype=np.float32)
        line_verts[0::2] = verts
        line_verts[1::2] = verts + normals * length

        colors = np.full((line_verts.shape[0], 3), 0.2, dtype=np.float32)
        data = np.hstack((line_verts, colors)).astype(np.float32)

        glBindVertexArray(self.vertex_normals_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_normals_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        return line_verts.shape[0]

    def handle_input(self):

        # Handle 'I' for toggling intersected faces
        i_state = glfw.get_key(self.window, glfw.KEY_I)
        if i_state == glfw.PRESS and self.last_i_state == glfw.RELEASE:
            self.show_intersected = not self.show_intersected
        self.last_i_state = i_state

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

        # Handle 'N' for per-face normals
        n_state = glfw.get_key(self.window, glfw.KEY_N)
        if n_state == glfw.PRESS and self.last_n_state == glfw.RELEASE:
            self.show_face_normals = not self.show_face_normals
        self.last_n_state = n_state

        # Handle 'M' for per-vertex normals
        m_state = glfw.get_key(self.window, glfw.KEY_M)
        if m_state == glfw.PRESS and self.last_m_state == glfw.RELEASE:
            self.show_vertex_normals = not self.show_vertex_normals
        self.last_m_state = m_state

        # Handle 'P' for point cloud
        p_state = glfw.get_key(self.window, glfw.KEY_P)
        if p_state == glfw.PRESS and self.last_p_state == glfw.RELEASE:
            self.show_point_cloud = not self.show_point_cloud
        self.last_p_state = p_state

        # Handle 'O' for Open
        o_state = glfw.get_key(self.window, glfw.KEY_O)
        if o_state == glfw.PRESS and self.last_o_state == glfw.RELEASE:
            self.open_file_dialog()
        self.last_o_state = o_state

        # Handle 'C' for Capture Screenshot
        c_state = glfw.get_key(self.window, glfw.KEY_C)
        if c_state == glfw.PRESS and self.last_c_state == glfw.RELEASE:
            self.capture_screenshot()
        self.last_c_state = c_state
    
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
        if self.show_intersected and self.intersected_index_count > 0:
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
        
        # --- DRAW PASS 3: Normals ---
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glUniform1i(self.use_override_loc, True)

        if self.show_face_normals and self.face_normals_count > 0:
            glBindVertexArray(self.face_normals_vao)
            glUniform3f(self.override_loc, 0.2, 0.8, 0.2) # Green
            glDrawArrays(GL_LINES, 0, self.face_normals_count)

        if self.show_vertex_normals and self.vertex_normals_count > 0:
            glBindVertexArray(self.vertex_normals_vao)
            glUniform3f(self.override_loc, 0.2, 0.6, 1.0) # Blue
            glDrawArrays(GL_LINES, 0, self.vertex_normals_count)
        
        if self.show_point_cloud and self.point_cloud_count > 0:
            glBindVertexArray(self.point_cloud_vao)
            glUniform3f(self.override_loc, 1.0, 1.0, 0.0) # Yellow
            glDrawArrays(GL_POINTS, 0, self.point_cloud_count)

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