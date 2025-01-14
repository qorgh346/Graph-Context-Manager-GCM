import graphviz


def visual_graph(data,scenario_id,mode='Test'):

    node_color_list = ['aliceblue', 'antiquewhite', 'cornsilk3', 'lightpink']
    obj_list = ['AMR_LIFT01','AMR_LIFT02','TOW_LIFT01','TOW_LIFT02']
    digraph1  = graphviz.Digraph(comment='The Scene Graph')

    for i, node in enumerate(obj_list):
        digraph1.attr('node', fillcolor=node_color_list[i], style='filled')
        digraph1.node(str(i), node)

    digraph1.attr('edge', fontname='Sans', color='black', style='filled')
    for i, edge in enumerate(data):
        temp_data = data[edge].split('_')
        source_node = temp_data[0]
        target_node = temp_data[1]
        edge_label = temp_data[2]

        digraph1.edge(str(source_node), str(target_node), str(edge_label))
    # save_graph_as_svg(digraph1,file_name)
    try:
        digraph1.render('./visualization/result_graph/visual_{}.gv'.format(scenario_id), view=True)  # 이름 바꾸기
    except:
        pass

