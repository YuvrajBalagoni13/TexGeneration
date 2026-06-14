bl_info = {
    "name":        "ShaderGen",
    "author":      "ShaderGen",
    "version":     (1, 0, 0),
    "blender":     (4, 0, 0),
    "location":    "Properties > Material > ShaderGen",
    "description": "Generate shader graphs from images using a VLM",
    "category":    "Material",
}

import bpy
import sys
import os
from bpy.props import (
    StringProperty,
    EnumProperty,
    BoolProperty,
    PointerProperty,
)
from bpy.types import Panel, Operator, PropertyGroup


class ShaderGenPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    env_site_packages: bpy.props.StringProperty(
        name="Site-Packages Path",
        default="",
        subtype='DIR_PATH'
    )

    def draw(self, context):
        self.layout.prop(self, "env_site_packages")
        self.layout.operator("shadergenv.reload_modules")


# ── Import your classes ─────────────────────────────────────────────────────
MODULES_LOADED = False
GGUFInference  = None
TextShader     = None

def load_modules():
    global MODULES_LOADED, GGUFInference, TextShader

    prefs    = bpy.context.preferences.addons[__name__].preferences
    env_path = prefs.env_site_packages.strip()

    # add to sys.path BEFORE importing gguf_inference
    if env_path and env_path not in sys.path:
        sys.path.append(env_path)
        print(f"[ShaderGen] Added: {env_path}")

    from .gguf_inference import GGUFInference
    from .txt_shader import TextShader

    MODULES_LOADED = True
    print("[ShaderGen] Modules loaded!")


class SHADERGENV_OT_reload_modules(bpy.types.Operator):
    bl_idname = "shadergenv.reload_modules"
    bl_label  = "Reload Modules"

    def execute(self, context):
        try:
            load_modules()
            self.report({'INFO'}, "Modules loaded!")
        except Exception as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}


# ── Precision options ───────────────────────────────────────────────────────
PRECISION_ITEMS = [
    ("F16",  "F16",  "float16 — highest quality, largest file",   0),
    ("Q8_0", "Q8",   "Q8_0 — near lossless, ~2x smaller",         1),
    ("Q5_K", "Q5",   "Q5_K_M — good balance of size and quality", 2),
    ("Q4_K", "Q4",   "Q4_K_M — smallest, fastest, slight loss",   3),
]

PRECISION_FILENAMES = {
    "F16":  "{model_name}_F16.gguf",
    "Q8_0": "{model_name}_Q8_0.gguf",
    "Q5_K": "{model_name}_Q5_K_M.gguf",
    "Q4_K": "{model_name}_Q4_K_M.gguf",
}


def get_model_path(props):
    filename = PRECISION_FILENAMES[props.precision].format(
        model_name=props.model_name
    )
    return os.path.join(bpy.path.abspath(props.model_dir), filename)

# ── Properties ──────────────────────────────────────────────────────────────

class ShaderGenProperties(PropertyGroup):
    image_path: StringProperty(
        name        = "Texture Image",
        description = "Input texture image to generate shader from",
        subtype     = "FILE_PATH",
        default     = ""
    )
    model_dir: StringProperty(
        name        = "Models Folder",
        description = "Folder containing your .gguf model files",
        subtype     = "DIR_PATH",
        default     = ""
    )
    model_name: StringProperty(
        name        = "Model Name",
        description = "Base name of the model (without precision suffix or .gguf)",
        default     = "Qwen3_5_0_8B_UT_TexGen"
    )
    precision: EnumProperty(
        name        = "Precision",
        description = "Model quantization precision",
        items       = PRECISION_ITEMS,
        default     = "Q8_0"
    )
    status_message: StringProperty(default="Ready")
    status_ok:      BoolProperty(default=True)
    last_dsl:       StringProperty(default="")


# ── Operators ───────────────────────────────────────────────────────────────


class SHADERGENV_OT_SetPrecision(Operator):
    bl_idname      = "shadergenv.set_precision"
    bl_label       = "Set Precision"
    bl_description = "Select model quantization precision"
    precision:      StringProperty()

    def execute(self, context):
        context.scene.shadergenv_props.precision = self.precision
        return {"FINISHED"}


class SHADERGENV_OT_Generate(Operator):
    bl_idname      = "shadergenv.generate"
    bl_label       = "Generate Shader"
    bl_description = "Run inference and apply the generated shader graph"

    def set_status(self, context, message, ok=True):
        props = context.scene.shadergenv_props
        props.status_message = message
        props.status_ok      = ok
        for area in context.screen.areas:
            area.tag_redraw()

    def execute(self, context):
        props = context.scene.shadergenv_props
        obj   = context.object
        mat   = obj.active_material if obj else None

        # ── Validate ────────────────────────────────────────
        if not MODULES_LOADED:
            self.report({"ERROR"}, f"Modules not loaded")
            return {"CANCELLED"}

        if not props.image_path:
            self.report({"ERROR"}, "No image selected")
            return {"CANCELLED"}

        image_path = bpy.path.abspath(props.image_path)
        if not os.path.isfile(image_path):
            self.report({"ERROR"}, f"Image not found: {image_path}")
            return {"CANCELLED"}

        if not props.model_dir:
            self.report({"ERROR"}, "No model folder selected")
            return {"CANCELLED"}

        # model_path = get_model_path(props)
        # if not os.path.isfile(model_path):
        #     self.report({"ERROR"}, f"Model not found: {os.path.basename(model_path)}")
        #     return {"CANCELLED"}

        # if mat is None:
        #     self.report({"ERROR"}, "No active material — create one first")
        #     return {"CANCELLED"}

        # ── Inference ───────────────────────────────────────
        self.set_status(context, "Running inference...")
        self.report({"INFO"}, f"Running inference ({props.precision})...")

        try:
            model_path = get_model_path(props)
            mmproj_path = os.path.join(props.model_dir, "mmproj_Qwen3_5_0_8B_UT_TexGen_F16.gguf")
            if os.path.isfile(model_path) and os.path.isfile(mmproj_path):
                inferencer = GGUFInference(
                    model_path=model_path,
                    mmproj_path=mmproj_path,
                    n_ctx = 1024,
                    max_tokens = 512,
                    temperature = 0.3,
                    top_p = 0.95
                )
            else:
                inferencer = GGUFInference(
                    model_repo="YuvrajB13/Qwen-3.5-0.8B-TexGen-GGUF",
                    model_file=PRECISION_FILENAMES[props.precision].format(model_name=props.model_name),
                    mmproj_file="mmproj_Qwen3_5_0_8B_UT_TexGen_F16.gguf",
                    n_ctx = 1024,
                    max_tokens = 512,
                    temperature = 0.3,
                    top_p = 0.95
                )
            dsl = inferencer.infer(image_path=props.image_path)
        except Exception as e:
            self.set_status(context, f"Inference error: {e}", ok=False)
            self.report({"ERROR"}, f"Inference failed: {e}")
            return {"CANCELLED"}

        if not dsl or not dsl.strip():
            self.set_status(context, "Model returned empty output", ok=False)
            self.report({"ERROR"}, "Model returned empty output")
            return {"CANCELLED"}

        props.last_dsl = dsl

        # ── Apply shader ────────────────────────────────────
        self.set_status(context, "Applying shader graph...")

        try:
            shader = TextShader()
            material = shader.text_to_shader_graph(
                text_shader=dsl,
                material_name = "generated_shader"
            )
            context.object.data.materials.append(material)
        except Exception as e:
            self.set_status(context, f"Shader error: {e}", ok=False)
            self.report({"ERROR"}, f"Shader apply failed: {e}")
            return {"CANCELLED"}

        self.set_status(context, "Done!", ok=True)
        self.report({"INFO"}, "Shader generated successfully")
        return {"FINISHED"}

# ── Panel ─────────────────────────────────────────────────────────────────────

class SHADERGENV_PT_Panel(Panel):
    bl_label       = "ShaderGen"
    bl_idname      = "SHADERGENV_PT_panel"
    bl_space_type  = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context     = "material"

    def draw(self, context):
        layout = self.layout
        props  = context.scene.shadergenv_props
        obj    = context.object
        mat    = obj.active_material if obj else None

        # ── Module warning ──────────────────────────────────
        if not MODULES_LOADED:
            box = layout.box()
            box.alert = True
            box.label(text="Could not import modules:", icon="ERROR")
            box.label(text="Update SHADERGENV_SRC_PATH in addon")
            layout.separator()

        # ── Material ────────────────────────────────────────
        # box = layout.box()
        # box.label(text="Material", icon="MATERIAL")
        # if mat:
        #     box.label(text=f"  {mat.name}", icon="CHECKMARK")
        # else:
        #     box.label(text="No material assigned", icon="ERROR")
        #     box.operator("shadergenv.new_material", icon="ADD")

        # layout.separator()

        # ── Image ───────────────────────────────────────────
        box = layout.box()
        box.label(text="Input Image", icon="IMAGE_DATA")
        box.prop(props, "image_path", text="")

        layout.separator()

        # ── Model ───────────────────────────────────────────
        box = layout.box()
        box.label(text="Model", icon="NETWORK_DRIVE")
        box.prop(props, "model_dir",  text="Folder - if installed model manually")
        box.prop(props, "model_name", text="Name")

        # Precision toggle buttons
        box.separator()
        box.label(text="Precision:")
        row = box.row(align=True)
        for value, label, _, _ in PRECISION_ITEMS:
            op = row.operator(
                "shadergenv.set_precision",
                text    = label,
                depress = (props.precision == value)
            )
            op.precision = value

        # Resolved filename preview
        if props.model_dir and props.model_name:
            fname = PRECISION_FILENAMES[props.precision].format(
                model_name=props.model_name
            )
            full_path = get_model_path(props)
            exists    = os.path.isfile(full_path)
            row = box.row()
            row.alert = not exists
            row.label(
                text = f"  {fname}",
                icon = "CHECKMARK" if exists else "ERROR"
            )

        layout.separator()

        # ── Generate button ─────────────────────────────────
        col = layout.column()
        col.scale_y = 1.8
        col.enabled = (
            MODULES_LOADED         and
            # mat   is not None      and
            bool(props.image_path) and
            bool(props.model_dir)  and
            bool(props.model_name)
        )
        col.operator("shadergenv.generate", icon="SHADERFX")

        # ── Status ──────────────────────────────────────────
        if props.status_message and props.status_message != "Ready":
            row = layout.row()
            row.alert = not props.status_ok
            icon = "CHECKMARK" if props.status_ok else "ERROR"
            row.label(text=props.status_message, icon=icon)

        # ── DSL actions ─────────────────────────────────────
        if props.last_dsl:
            layout.separator()
            box = layout.box()
            box.label(text="Last Output", icon="TEXT")
            row = box.row()
            # row.operator("shadergenv.reapply",  icon="FILE_REFRESH")
            # row.operator("shadergenv.copy_dsl", icon="COPYDOWN")


# ── Registration ──────────────────────────────────────────────────────────────

CLASSES = [
    ShaderGenProperties,
    SHADERGENV_OT_SetPrecision,
    SHADERGENV_OT_Generate,
    SHADERGENV_PT_Panel,
]


def register():
    bpy.utils.register_class(ShaderGenPreferences)
    bpy.utils.register_class(SHADERGENV_OT_reload_modules)
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.shadergenv_props = PointerProperty(type=ShaderGenProperties)


def unregister():
    bpy.utils.unregister_class(ShaderGenPreferences)
    bpy.utils.unregister_class(SHADERGENV_OT_reload_modules)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.shadergenv_props


if __name__ == "__main__":
    register()