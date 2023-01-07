# This script is modified from https://github.com/isl-org/Open3D/blob/master/examples/python/gui/vis-gui.py

# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------
import os
import sys
import copy
import glob
import torch
import joblib
import platform
import numpy as np
import open3d as o3d
from loguru import logger
import open3d.visualization.gui as gui
import scipy.spatial.transform.rotation as R
import open3d.visualization.rendering as rendering


from utils import (
    get_checkerboard_plane,
    smpl_joint_names,
    smplx_body_joint_names,
    hand_joint_names,
    LEFT_HAND_KEYPOINT_NAMES,
    RIGHT_HAND_KEYPOINT_NAMES,
    HEAD_KEYPOINT_NAMES,
    FLAME_KEYPOINT_NAMES,
    FOOT_KEYPOINT_NAMES,
    SMPL_NAMES,
    SMPLX_NAMES,
    MANO_NAMES,
)
from simple_ik import simple_ik_solver


isMacOS = (platform.system() == "Darwin")


class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = False
        self.show_axes = True
        self.show_ground = True
        self.use_ibl = True
        self.use_sun = True
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 45000
        self.sun_intensity = 45000
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.LIT]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True



    def apply_lighting_profile(self, name):
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SAVE = 4
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH
    ]

    BODY_MODEL_NAMES = ["SMPL", "SMPLX", "MANO", "FLAME"]
    BODY_MODEL_GENDERS = {
        'SMPL': ['neutral', 'male', 'female'],
        'SMPLX': ['neutral', 'male', 'female'],
        'MANO': ['neutral'],
        'FLAME': ['neutral', 'male', 'female']
    }
    BODY_MODEL_N_BETAS = {
        'SMPL': 10,
        'SMPLX': 10,
        'MANO': 10,
        'FLAME': 10,
    }
    CAM_FIRST = True

    PRELOADED_BODY_MODELS = {}

    POSE_PARAMS = {
        'SMPL': {
            'body_pose': torch.zeros(1, 23, 3),
            'global_orient': torch.zeros(1, 1, 3),
        },
        'SMPLX': {
            'body_pose': torch.zeros(1, 21, 3),
            'global_orient': torch.zeros(1, 1, 3),
            'left_hand_pose': torch.zeros(1, 15, 3),
            'right_hand_pose': torch.zeros(1, 15, 3),
            'jaw_pose': torch.zeros(1, 1, 3),
            'leye_pose': torch.zeros(1, 1, 3),
            'reye_pose': torch.zeros(1, 1, 3),
        },
        'MANO': {
            'hand_pose': torch.zeros(1, 15, 3),
            'global_orient': torch.zeros(1, 1, 3),
        },
        'FLAME': {
            'global_orient': torch.zeros(1, 1, 3),
            'jaw_pose': torch.zeros(1, 1, 3),
            'neck_pose': torch.zeros(1, 1, 3),
            'leye_pose': torch.zeros(1, 1, 3),
            'reye_pose': torch.zeros(1, 1, 3),
        },
    }

    JOINT_NAMES = {
        'SMPL': {
            'global_orient': ['root'],
            'body_pose': smpl_joint_names,
        },
        'SMPLX': {
            'global_orient': ['root'],
            'body_pose': smplx_body_joint_names,
            'left_hand_pose': hand_joint_names,
            'right_hand_pose': hand_joint_names,
            'jaw_pose': ['jaw'],
            'leye_pose': ['leye'],
            'reye_pose': ['reye'],
        },
        'MANO': {
            'global_orient': ['root'],
            'hand_pose': hand_joint_names,
        },
        'FLAME': {
            'global_orient': ['root'],
            'jaw_pose': ['jaw'],
            'neck_pose': ['neck'],
            'leye_pose': ['leye'],
            'reye_pose': ['reye'],
        },
    }

    KEYPOINT_NAMES = {
        'SMPL': SMPL_NAMES,
        'SMPLX': SMPLX_NAMES,
        'MANO': MANO_NAMES,
        'FLAME': FLAME_KEYPOINT_NAMES,
    }

    JOINTS = None
    SELECTED_JOINT = None
    BODY_TRANSL = None

    def __init__(self, width, height):
        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.window = gui.Application.instance.create_window(
            "Open3D", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)


        em = w.theme.font_size
        self.em = em
        separation_height = int(round(0.5 * em))


        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))



        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)

        self._settings_panel.add_fixed(separation_height)



        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)

        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)


        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path +
                             "/*_ibl.ktx"):

            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = AppWindow.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)

        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)


        self._settings_panel.add_fixed(separation_height)





        self._settings_panel.add_fixed(separation_height)


        # ----------------------------------- #
        # ------- BODY MODEL SETTINGS ------- #
        # ----------------------------------- #
        self.preload_body_models()
        self._scene.scene.show_ground_plane(self.settings.show_ground, rendering.Scene.GroundPlane(0))
        self.model_settings = gui.CollapsableVert("Model settings", 0,
                                                  gui.Margins(em, 0, 0, 0))
        self.model_settings.set_is_open(True)

        self._body_model = gui.Combobox()
        for bm in AppWindow.BODY_MODEL_NAMES:
            self._body_model.add_item(bm)

        self._body_model_gender = gui.Combobox()
        for gender in AppWindow.BODY_MODEL_GENDERS[AppWindow.BODY_MODEL_NAMES[0]]:
            self._body_model_gender.add_item(gender)

        # ------- BODY MODEL BETAS SETTINGS ------- #
        self._body_model_shape_comp = gui.Combobox()
        for i in range(AppWindow.BODY_MODEL_N_BETAS[AppWindow.BODY_MODEL_NAMES[0]]):
            self._body_model_shape_comp.add_item(f'{i+1:02d}')

        self._body_beta_val = gui.Slider(gui.Slider.DOUBLE)
        self._body_beta_val.set_limits(-5.0, 5.0)
        self._body_beta_tensor = torch.zeros(1, 10)
        self._body_beta_reset = gui.Button("Reset betas")

        self._body_beta_text = gui.Label("Betas")
        self._body_beta_text.text = f",".join(f'{x:.1f}'for x in self._body_beta_tensor[0].numpy().tolist())

        # ------- BODY MODEL EXPRESSION SETTINGS ------- #
        self._body_model_exp_comp = gui.Combobox()
        for i in range(10):
            self._body_model_exp_comp.add_item(f'{i + 1:02d}')

        self._body_exp_val = gui.Slider(gui.Slider.DOUBLE)
        self._body_exp_val.set_limits(-5.0, 5.0)
        self._body_exp_tensor = torch.zeros(1, 10)
        self._body_exp_reset = gui.Button("Reset expression")

        self._body_exp_text = gui.Label("Expression")
        self._body_exp_text.text = f",".join(f'{x:.1f}' for x in self._body_exp_tensor[0].numpy().tolist())




        self._body_beta_val.set_on_value_changed(self._on_body_beta_val)
        self._body_beta_reset.set_on_clicked(self._on_body_beta_reset)
        self._body_model_shape_comp.set_on_selection_changed(self._on_body_model_shape_comp)

        self._body_exp_val.set_on_value_changed(self._on_body_exp_val)
        self._body_exp_reset.set_on_clicked(self._on_body_exp_reset)
        self._body_model_exp_comp.set_on_selection_changed(self._on_body_model_exp_comp)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Body Model"))
        grid.add_child(self._body_model)
        grid.add_child(gui.Label("Gender"))
        grid.add_child(self._body_model_gender)
        grid.add_child(gui.Label("Beta Component"))
        grid.add_child(self._body_model_shape_comp)
        grid.add_child(gui.Label("Beta val:"))
        grid.add_child(self._body_beta_val)
        self.model_settings.add_child(grid)



        h = gui.Horiz(0.25 * em)  # row 2
        h.add_child(self._body_beta_reset)
        self.model_settings.add_child(h)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Exp Component"))
        grid.add_child(self._body_model_exp_comp)
        grid.add_child(gui.Label("Exp val:"))
        grid.add_child(self._body_exp_val)
        self.model_settings.add_child(grid)

        h = gui.Horiz(0.25 * em)  # row 2
        h.add_child(self._body_exp_reset)
        self.model_settings.add_child(h)

        h = gui.Horiz(0.25 * em)  # row 2

        self.model_settings.add_child(h)

        h = gui.Horiz(0.25 * em)  # row 3


        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self.model_settings)

        # Info panel
        self.info = gui.Label("")
        self.info.visible = False

        self.joint_label_3d = gui.Label3D("", [0,0,0])
        self.joint_labels_3d_list = []

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        w.add_child(self.info)

        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("Help", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image", AppWindow.MENU_EXPORT)
            file_menu.add_item("Save Model Params", AppWindow.MENU_SAVE)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Model - Lighting - Materials",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        #w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_SAVE,
                                     self._on_save_dialog)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        self._apply_settings()

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False



        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])


    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

        pref = self.info.calc_preferred_size(layout_context,
                                             gui.Widget.Constraints())
        self.info.frame = gui.Rect(r.x,
                                   r.get_bottom() - pref.height, pref.width,
                                   pref.height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _set_mouse_mode_pick(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.PICK_POINTS)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_show_ground(self, show):
        self.settings.show_ground = show
        self._apply_settings()

    def _on_show_joint_labels(self, show):
        if hasattr(self, "joint_label_3d"):
            self._scene.remove_3d_label(self.joint_label_3d)
        if hasattr(self, "joint_labels_3d_list"):
            for label3d in self.joint_labels_3d_list:
                self._scene.remove_3d_label(label3d)
        if show:
            joint_names = AppWindow.KEYPOINT_NAMES[self._body_model.selected_text]
            try:
                for i in range(len(joint_names)):
                    self.joint_labels_3d_list.append(
                        self._scene.add_3d_label(AppWindow.JOINTS[i], joint_names[i])
                    )
            except Exception as e:
                print(e)
                import ipdb; ipdb.set_trace()
        else:
            if hasattr(self, "joint_labels_3d_list"):
                for label3d in self.joint_labels_3d_list:
                    self._scene.remove_3d_label(label3d)



        green = [0.3, 0.7, 0.3, 1.0]
        red = [0.7, 0.3, 0.3, 1.0]
        hand_radius = 0.01
        foot_radius = 0.01
        head_radius = 0.007
        body_radius = 0.05
        joint_names = AppWindow.KEYPOINT_NAMES[self._body_model.selected_text]

        mat = rendering.MaterialRecord()
        mat.base_color = red
        mat.shader = "defaultLit"

        mat_selected = rendering.MaterialRecord()
        mat_selected.base_color = green
        mat_selected.shader = "defaultLit"

        joints = AppWindow.JOINTS
        if show:
            # logger.info('drawing joints')
            for i in range(joints.shape[0]):
                radius = body_radius
                if joint_names[i] in LEFT_HAND_KEYPOINT_NAMES + RIGHT_HAND_KEYPOINT_NAMES:
                    radius = hand_radius
                elif joint_names[i] in HEAD_KEYPOINT_NAMES:
                    radius = head_radius
                elif joint_names[i] in FOOT_KEYPOINT_NAMES:
                    radius = foot_radius

                sg = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sg.compute_vertex_normals()
                # if i == AppWindow.SELECTED_JOINT:
                #     sg.paint_uniform_color(green)
                # else:
                #     sg.paint_uniform_color(red)
                sg.translate(joints[i])
                if (AppWindow.SELECTED_JOINT is not None) and (i == AppWindow.SELECTED_JOINT):
                    self._scene.scene.add_geometry(f"__joints_{i}__", sg, mat_selected)
                else:
                    self._scene.scene.add_geometry(f"__joints_{i}__", sg, mat)

            # logger.debug(AppWindow.JOINTS[20])
        else:
            # import ipdb; ipdb.set_trace()
            for i in range(150):
                if self._scene.scene.has_geometry(f"__joints_{i}__"):
                    self._scene.scene.remove_geometry(f"__joints_{i}__")

        self._on_show_joint_labels(self._show_joint_labels.checked)
        # import ipdb; ipdb.set_trace()

    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name, index):
        self.settings.new_ibl_name = gui.Application.instance.resource_path + "/" + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_body_model(self, name, index):
        logger.info(f"Loading body model {name}-{index}")
        self._body_beta_val.double_value = 0.0
        AppWindow.CAM_FIRST = True
        self.load_body_model(name)
        self._body_model_gender.clear_items()

        for gender in AppWindow.BODY_MODEL_GENDERS[name]:
            self._body_model_gender.add_item(gender)



        self._reset_rot_sliders()
        AppWindow.SELECTED_JOINT = None


    def _on_body_model_gender(self, name, index):
        logger.info(f"Changing {self._body_model.selected_text} body model gender to {name}-{index}")
        self._body_beta_val.double_value = 0.0
        self.load_body_model(self._body_model.selected_text, gender=name)
        self._reset_rot_sliders()
        # self._on_show_joints(self._show_joints.checked)
        # self._apply_settings()

    def _on_body_beta_val(self, val):
        self._body_beta_tensor[0, int(self._body_model_shape_comp.selected_text)-1] = float(val)
        self._body_beta_text.text = f",".join(f'{x:.1f}' for x in self._body_beta_tensor[0].numpy().tolist())
        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )
        # self._on_show_joints(self._show_joints.checked)

    def _on_body_exp_val(self, val):
        self._body_exp_tensor[0, int(self._body_model_exp_comp.selected_text)-1] = float(val)
        self._body_exp_text.text = f",".join(f'{x:.1f}' for x in self._body_exp_tensor[0].numpy().tolist())
        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )
        # self._on_show_joints(self._show_joints.checked)

    def _on_body_pose_joint(self, name, index):
        self._reset_rot_sliders()

    def _on_body_pose_joint_x(self, val):
        bm = self._body_model.selected_text
        bp = self._body_pose_comp.selected_text
        ji = int(self._body_pose_joint.selected_text.split('-')[0])
        euler_angle = [val, self._body_pose_joint_y.int_value, self._body_pose_joint_z.int_value]
        axis_angle = R.Rotation.from_euler('xyz', euler_angle, degrees=True).as_rotvec()
        AppWindow.POSE_PARAMS[bm][bp][0, ji] = torch.from_numpy(axis_angle)

        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )
        # self._on_show_joints(self._show_joints.checked)

    def _on_body_pose_joint_y(self, val):
        bm = self._body_model.selected_text
        bp = self._body_pose_comp.selected_text
        ji = int(self._body_pose_joint.selected_text.split('-')[0])
        euler_angle = [self._body_pose_joint_x.int_value, val, self._body_pose_joint_z.int_value]
        axis_angle = R.Rotation.from_euler('xyz', euler_angle, degrees=True).as_rotvec()
        AppWindow.POSE_PARAMS[bm][bp][0, ji] = torch.from_numpy(axis_angle)

        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )
        # self._on_show_joints(self._show_joints.checked)

    def _on_body_pose_joint_z(self, val):
        bm = self._body_model.selected_text
        bp = self._body_pose_comp.selected_text
        ji = int(self._body_pose_joint.selected_text.split('-')[0])
        euler_angle = [self._body_pose_joint_x.int_value, self._body_pose_joint_y.int_value, val]
        axis_angle = R.Rotation.from_euler('xyz', euler_angle, degrees=True).as_rotvec()
        AppWindow.POSE_PARAMS[bm][bp][0, ji] = torch.from_numpy(axis_angle)

        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )
        # self._on_show_joints(self._show_joints.checked)

    def _on_body_model_shape_comp(self, name, index):
        self._body_beta_val.double_value = self._body_beta_tensor[0, index].item()

    def _on_body_model_exp_comp(self, name, index):
        self._body_exp_val.double_value = self._body_exp_tensor[0, index].item()

    def _on_body_pose_comp(self, name, index):
        self._body_pose_joint.clear_items()
        joint_names = AppWindow.JOINT_NAMES[self._body_model.selected_text][name]
        for i in range(AppWindow.POSE_PARAMS[self._body_model.selected_text][name].shape[1]):
            self._body_pose_joint.add_item(f'{i}-{joint_names[i]}')
        self._reset_rot_sliders()

    def _on_body_beta_reset(self):
        self._body_beta_tensor = torch.zeros(1, 10)
        self._body_beta_text.text = f",".join(f'{x:.1f}' for x in self._body_beta_tensor[0].numpy().tolist())
        self._body_beta_val.double_value = 0.0
        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )

    def _on_body_exp_reset(self):
        self._body_exp_tensor = torch.zeros(1, 10)
        self._body_exp_text.text = f",".join(f'{x:.1f}' for x in self._body_exp_tensor[0].numpy().tolist())
        self._body_exp_val.double_value = 0.0
        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )

    def _on_body_pose_reset(self):
        bm = self._body_model.selected_text
        bp = self._body_pose_comp.selected_text
        AppWindow.POSE_PARAMS[bm][bp] = torch.zeros_like(AppWindow.POSE_PARAMS[bm][bp])
        self._reset_rot_sliders()
        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )



    def _update_label(self, text):
        self.info.text = text
        self.info.visible = (text != "")
        # We are sizing the info label to be exactly the right size,
        # so since the text likely changed width, we need to
        # re-layout to set the new frame.
        self.window.set_needs_layout()

    def _on_run_ik(self):
        bm = self._body_model.selected_text
        bp = self._body_pose_comp.selected_text

        if not ((bm in ['SMPL', 'SMPLX']) and (bp in ('body_pose'))):
            logger.warning('IK is not implemented for this body model')
            return 0

        gender = self._body_model_gender.selected_text
        init_pose = copy.deepcopy(AppWindow.POSE_PARAMS[bm][bp])

        target_keypoints = AppWindow.JOINTS[:22][None]
        target_keypoints = torch.from_numpy(target_keypoints).float()
        opt_params = simple_ik_solver(
            model=AppWindow.PRELOADED_BODY_MODELS[f'{bm.lower()}-{gender.lower()}'],
            target=target_keypoints, init=init_pose, device='cpu',
            max_iter=50, transl=AppWindow.BODY_TRANSL,
            betas=self._body_beta_tensor,
        )
        opt_params = opt_params.requires_grad_(False)
        # import ipdb; ipdb.set_trace()
        AppWindow.POSE_PARAMS[bm][bp] = opt_params.reshape(1, -1, 3)

        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )

    def _reset_rot_sliders(self):
        self._body_pose_joint_x.int_value = 0
        self._body_pose_joint_y.int_value = 0
        self._body_pose_joint_z.int_value = 0






    def _on_save_dialog(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.set_on_cancel(self._on_save_dialog_cancel)
        dlg.set_on_done(self._on_save_dialog_done)
        self.window.show_dialog(dlg)

    def _on_save_dialog_cancel(self):
        self.window.close_dialog()

    def _on_save_dialog_done(self, filename):
        self.window.close_dialog()
        output_dict = {
            'betas': self._body_beta_tensor,
            'expression': self._body_exp_tensor,
            'gender': self._body_model_gender.selected_text,
            'body_model': self._body_model.selected_text,
            'joints': AppWindow.JOINTS,
        }
        output_dict.update(AppWindow.POSE_PARAMS[self._body_model.selected_text])
        logger.debug(f'Saving output to {filename}')
        joblib.dump(output_dict, filename)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Body Model Visualizer - Help"))
        dlg_layout.add_child(gui.Label("Select joint: Ctrl+left click"))
        dlg_layout.add_child(gui.Label("-- Move selected Joint --"))
        dlg_layout.add_child(gui.Label("Move -x/+x: 1/2"))
        dlg_layout.add_child(gui.Label("Move -y/+y: 3/4"))
        dlg_layout.add_child(gui.Label("Move -z/+z: 5/6"))
        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def add_ground_plane(self):
        logger.info('drawing ground plane')
        gp = get_checkerboard_plane(plane_width=2, num_boxes=9)

        for idx, g in enumerate(gp):
            g.compute_vertex_normals()
            self._scene.scene.add_geometry(f"__ground_{idx:04d}__", g, self.settings._materials[Settings.LIT])

    def preload_body_models(self):
        from smplx import SMPL, SMPLX, MANO, FLAME

        for body_model in AppWindow.BODY_MODEL_NAMES:
            for gender in AppWindow.BODY_MODEL_GENDERS[body_model]:
                logger.info(f'Loading {body_model}-{gender}')
                extra_params = {'gender': gender}
                if body_model in ('SMPLX', 'MANO', 'FLAME'):
                    extra_params['use_pca'] = False
                    extra_params['flat_hand_mean'] = True
                    extra_params['use_face_contour'] = True
                model = eval(body_model.upper())(f'data/body_models/{body_model.lower()}', **extra_params)
                AppWindow.PRELOADED_BODY_MODELS[f'{body_model.lower()}-{gender.lower()}'] = model
        logger.info(f'Loaded body models {AppWindow.PRELOADED_BODY_MODELS.keys()}')

    # @torch.no_grad()
    def load_body_model(self, body_model='smpl', gender='neutral'):
        self._scene.scene.remove_geometry("__body_model__")

        model = AppWindow.PRELOADED_BODY_MODELS[f'{body_model.lower()}-{gender.lower()}']

        input_params = copy.deepcopy(AppWindow.POSE_PARAMS[body_model])

        for k, v in input_params.items():
            input_params[k] = v.reshape(1, -1)

        model_output = model(
            betas=self._body_beta_tensor,
            expression=self._body_exp_tensor,
            **input_params,
        )
        verts = model_output.vertices[0].detach().numpy()
        AppWindow.JOINTS = model_output.joints[0].detach().numpy()
        faces = model.faces

        mesh = o3d.geometry.TriangleMesh()

        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
        # ipdb.set_trace()
        min_y = -mesh.get_min_bound()[1]
        mesh.translate([0, min_y, 0])
        AppWindow.JOINTS += np.array([0, min_y, 0])

        self._scene.scene.add_geometry("__body_model__", mesh,
                                       self.settings.material)
        bounds = mesh.get_axis_aligned_bounding_box()
        if AppWindow.CAM_FIRST:
            self._scene.setup_camera(60, bounds, bounds.get_center())
            AppWindow.CAM_FIRST = False
        AppWindow.BODY_TRANSL = torch.tensor([[0, min_y, 0]])
        #self._on_show_joints(self._show_joints.checked)

    def load(self, path):
        # self._scene.scene.clear_geometry()
        # if self.settings.show_ground:
        #     self.add_ground_plane()

        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_mesh(path)
        if mesh is not None:
            if len(mesh.triangles) == 0:
                print(
                    "[WARNING] Contains 0 triangles, will read as point cloud")
                mesh = None
            else:
                mesh.compute_vertex_normals()
                if len(mesh.vertex_colors) == 0:
                    mesh.paint_uniform_color([1, 1, 1])
                geometry = mesh
            # Make sure the mesh has texture coordinates
            if not mesh.has_triangle_uvs():
                uv = np.array([[0.0, 0.0]] * (3 * len(mesh.triangles)))
                mesh.triangle_uvs = o3d.utility.Vector2dVector(uv)
        else:
            print("[Info]", path, "appears to be a point cloud")

        if geometry is None:
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)

        if geometry is not None:
            try:
                self._scene.scene.add_geometry("__model__", geometry,
                                               self.settings.material)
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)


def main():
    # We need to initalize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1920, 1080)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
