 # Adds Lighting
        world = bpy.context.scene.world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()

        world_out = nodes.new(type='ShaderNodeOutputWorld')
        bg_node = nodes.new(type='ShaderNodeBackground')
        sky_node = nodes.new(type='ShaderNodeTexSky')

        sky_node.sky_type = 'HOSEK_WILKIE'

        sky_node.sun_elevation = 1.57
        sky_node.sun_rotation = 0.0
        sky_node.sun_intensity = 0.8

        bg_node.inputs['Strength'].default_value = 0.05

        links.new(sky_node.outputs['Color'], bg_node.inputs['Color'])
        links.new(bg_node.outputs['Background'], world_out.inputs['Surface'])
        
        light_data = bpy.data.lights.new(name="Sun_Data", type='SUN')
        light_data.energy = 0.7
        light_data.angle = 0.0
        light_obj = bpy.data.objects.new(name="Sun_Light", object_data=light_data)
        bpy.context.scene.collection.objects.link(light_obj)
        light_obj.location = (0, 0, 10)
        light_obj.rotation_euler = (0.0,0.0,0.0)