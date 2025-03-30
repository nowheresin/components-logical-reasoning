import csv
import copy
from utilities import *

# DEBUG = True
# _print = print
# print = lambda *args, **kwargs: _print(*args, **kwargs) if DEBUG else ...

from logLEVEL import print
print.LEVEL = 'DEBUG'

def write_in_csv(file_name, data):
    """
    向csv文件中写入数据
    :param file_name: 文件名
    :param data: [
    ['Name', 'Age', 'City'],
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'San Francisco'],
    ['Charlie', 35, 'Los Angeles']
    ]
    :return:
    """
    # 打开CSV文件，将数据写入
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

    print("CSV文件已创建并写入数据。", LOG='INFO')

def get_from_csv(file_name):
    """
    从csv文件中获得数据
    :param file_name: 文件名
    :return: 数据
    """
    data = []
    # 打开CSV文件
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        # 循环读取每一行内容
        for row in reader:
            # print(row, LOG='DEBUG')
            data.append(row)

    return data

def get_database(data, startLine=1):
    """
    从数据中获得数据集
    :param data: 数据
    :param startLine: 起始行数
    :return: 数据集
    """
    database = {}

    for item in data[startLine:]:
        one_component_info = item[0].split('; ')
        # print(one_component_info, LOG='DEBUG')
        product_info, disassembly_seq, component_type, geometry, colour, fastener_type, isEnd = [x.split(', ') for x in
                                                                                                 one_component_info]

        if product_info[0] not in database.keys():
            database[product_info[0]] = {'product_type': product_info[1]}
        elif database[product_info[0]]['product_type'] != product_info[1]:
            raise ValueError('product type is not right!')

        if int(disassembly_seq[0]) not in database[product_info[0]].keys():
            database[product_info[0]][int(disassembly_seq[0])] = {'component_type': component_type}
            database[product_info[0]][int(disassembly_seq[0])]['geometry'] = geometry
            database[product_info[0]][int(disassembly_seq[0])]['colour'] = colour
            database[product_info[0]][int(disassembly_seq[0])]['fastener_type'] = [fastener_type[i:i + 2] for i in
                                                                                        range(0, len(fastener_type), 2)]
            database[product_info[0]][int(disassembly_seq[0])]['isEnd'] = [eval(isEnd[0])]
        else:
            raise ValueError('disassembly sequence is not right!')

        # print(database, LOG='DEBUG')
    print('database:', database, LOG='DEBUG')
    return database

def get_component_num(database):
    """
    从数据集中获得零部件数量和产品数量
    :param database: 数据集
    :return: 零部件数量和产品数量
    """
    component_num = {}
    product_num = 0
    for product in database.values():
        for item in product.values():
            # print(item, LOG='DEBUG')
            if isinstance(item, dict) and 'component_type' in item.keys():
                component_name = item['component_type'][0]
                if component_name not in component_num.keys():
                    component_num[component_name] = 1
                else:
                    component_num[component_name] += 1
        product_num += 1

    return component_num, product_num

def get_products(database):
    """
    获取数据集中的各类产品类型
    :param database: 数据集
    :return: 产品列表
    """
    products = []
    for product in database.values():
        product_name = product['product_type']
        if product_name not in products:
            products.append(product_name)
    print('products:', products, LOG='INFO')

    return products


def get_common_components(database, threshold=0.3):
    """
    获取常见零部件列表
    :param database: 数据集
    :param threshold: 阈值
    :return: 常见零部件列表
    """

    component_num, product_num = get_component_num(database)
    # print(component_num, LOG='DEBUG')
    # print(product_num, LOG='DEBUG')

    common_components = []
    for component_name, num in component_num.items():
        if num / product_num > threshold:
            common_components.append(component_name)
    print('common_components:', common_components, LOG='INFO')

    return common_components

def get_preceding_and_immediately_components(database, common_components, threshold=0.5):
    """
    获取preceding和immediately零部件数据
    :param database: 数据集
    :param common_components: 常见零部件列表
    :return: preceding和immediately零部件数据
    """
    preceding_components, immediately_preceding_component = {}, {}

    # 含有product_type的字典
    DisassemblyState_with_product_type = {
        ID: {k: v['component_type'][0] if isinstance(v, dict) and 'component_type' in v.keys() else v for k, v in
             product.items()} for ID, product in database.items()}
    # print(DisassemblyState_with_product_type, LOG='DEBUG')

    DisassemblyState = {ID: {k: v for k, v in product.items() if isinstance(k, int)} for ID, product in
                        DisassemblyState_with_product_type.items()}

    # print(DisassemblyState, LOG='DEBUG')
    for ID, product in DisassemblyState.items():
        if ID not in preceding_components.keys():
            preceding_components[ID] = {}
            immediately_preceding_component[ID] = {}
        for state, component in product.items():
            if component in common_components:
                preceding_components[ID].setdefault(component, [])
                immediately_preceding_component[ID].setdefault(component, [])

                for other_state, other_component in product.items():
                    if other_state < state:
                        preceding_components[ID][component].append(other_component)
                    if other_state == state - 1:
                        immediately_preceding_component[ID][component].append(other_component)
    # print(preceding_components, LOG='DEBUG')
    # print(immediately_preceding_component, LOG='DEBUG')

    immediately_preceding = {}
    for item in immediately_preceding_component.values():
        for component, preceding in item.items():
            immediately_preceding.setdefault(component, preceding)
            for immediately_component in preceding:
                if immediately_component not in immediately_preceding[component]:
                    immediately_preceding[component].append(immediately_component)

    # 移除一些可能异常的零部件
    component_num = {}
    for item in preceding_components.values():
        for component in item.keys():
            if component not in component_num.keys():
                component_num[component] = 1
            else:
                component_num[component] += 1
    # print(component_num, LOG='DEBUG')

    removal_components = []
    for component, num in component_num.items():
        for other_component, other_num in component_num.items():
            if other_num / num < threshold:
                if other_component not in removal_components:
                    removal_components.append(other_component)
    # print(removal_components, LOG='DEBUG')

    for component in removal_components:
        for item in preceding_components.values():
            item.pop(component)
            for v in item.values():
                v.remove(component)

    print('preceding_components:', preceding_components, LOG='INFO')
    # print('immediately_preceding_component:', immediately_preceding_component, LOG='INFO')
    print('immediately_preceding:', immediately_preceding, LOG='INFO')

    return preceding_components, immediately_preceding

def get_larger_and_smaller_than(database, common_components, preceding_components=None):
    """
    获取满足尺寸条件的数据
    :param database: 数据集
    :param common_components: 常见零部件数据列表
    :param preceding_components: preceding零部件数据
    :return: smallest_larger_than和largest_smaller_than
    """
    ini_component = preceding_components if preceding_components is not None else {ID: {
        component['component_type'][0]: [other_component['component_type'][0] for other_component in product.values() if
                                         isinstance(other_component, dict) and other_component is not component] for
        component in product.values() if isinstance(component, dict)} for ID, product in database.items()}
    # print(ini_component, LOG='DEBUG')
    larger_than, smaller_than = copy.deepcopy(ini_component), copy.deepcopy(ini_component)

    ini_size = {ID: {component['component_type'][0]: int(component['geometry'][2]) for component in product.values() if
                     isinstance(component, dict)} for ID, product in database.items()}
    # print(ini_size, LOG='DEBUG')
    for ID, product in ini_size.items():
        for component, size in product.items():
            if component in common_components:
                for other_component, other_size in product.items():
                    if other_component in common_components and component is not other_component:
                        if size > other_size and other_component in larger_than[ID][component]:
                            larger_than[ID][component].remove(other_component)
                        elif size <= other_size and other_component in larger_than[ID][component]:
                            smaller_than[ID][component].remove(other_component)
    # print('larger_than:', larger_than, LOG='DEBUG')
    # print('smaller_than:', smaller_than, LOG='DEBUG')

    smallest_larger_than_in_product = copy.deepcopy(larger_than)
    for ID, product in smallest_larger_than_in_product.items():
        for component, than in product.items():
            if than == []:
                product[component].append(ini_size[ID][component])
            else:
                compare_list = product[component]
                compare_list_num = [ini_size[ID][item] for item in compare_list]
                # print(compare_list, LOG='DEBUG')
                min_index = min(enumerate(compare_list_num), key=lambda x: x[1])[0]
                # print(min_index, LOG='DEBUG')
                product[component] = [compare_list[min_index]]

    smallest_larger_than = {}
    for item in smallest_larger_than_in_product.values():
        for component, than in item.items():
            smallest_larger_than.setdefault(component, than)
            if isinstance(than[0], int) and isinstance(smallest_larger_than[component][0], int) and than[0] < smallest_larger_than[component][0]:
                smallest_larger_than[component][0] = than[0]
            elif isinstance(than[0], str) and than[0] not in smallest_larger_than[component]:
                smallest_larger_than[component].append(than[0])
            elif isinstance(than[0], str) and isinstance(smallest_larger_than[component][0], int):
                smallest_larger_than[component][0] = than[0]
    print('smallest_larger_than:', smallest_larger_than, LOG='INFO')

    largest_smaller_than_in_product = copy.deepcopy(smaller_than)
    for ID, product in largest_smaller_than_in_product.items():
        for component, than in product.items():
            if than == []:
                product[component].append(ini_size[ID][component])
            else:
                compare_list = product[component]
                compare_list_num = [ini_size[ID][item] for item in compare_list]
                # print(compare_list_num, LOG='DEBUG')
                max_index = max(enumerate(compare_list_num), key=lambda x: x[1])[0]
                # print(max_index, LOG='DEBUG')
                product[component] = [compare_list[max_index]]

    largest_smaller_than = {}
    for item in largest_smaller_than_in_product.values():
        for component, than in item.items():
            largest_smaller_than.setdefault(component, than)
            if isinstance(than[0], int) and isinstance(largest_smaller_than[component][0], int) and than[0] > largest_smaller_than[component][0]:
                largest_smaller_than[component][0] = than[0]
            elif isinstance(than[0], str) and than[0] not in largest_smaller_than[component]:
                largest_smaller_than[component].append(than[0])
            elif isinstance(than[0], str) and isinstance(largest_smaller_than[component][0], int):
                largest_smaller_than[component][0] = than[0]
    print('largest_smaller_than:', largest_smaller_than, LOG='INFO')

    return smallest_larger_than, largest_smaller_than

def get_component_common_geometry(database):
    """
    获得每个零部件的常见几何形状
    :param database: 数据集
    :return: 零部件的常见几何形状
    """
    component_geometry = {}
    for product in database.values():
        for item in product.values():
            if isinstance(item, dict):
                component_name = item['component_type'][0]
                geometry = item['geometry'][0]
                component_geometry.setdefault(component_name, {})
                if geometry not in component_geometry[component_name]:
                    component_geometry[component_name][geometry] = 1
                else:
                    component_geometry[component_name][geometry] += 1

    component_common_geometry = {}
    for component_name, geometries in component_geometry.items():
        geometries_num = [num for num in geometries.values()]
        geometries_name = [name for name in geometries.keys()]
        max_index = max(enumerate(geometries_num), key=lambda x: x[1])[0]

        component_common_geometry[component_name] = geometries_name[max_index]

    print('component_common_geometry:', component_common_geometry, LOG='INFO')
    return component_common_geometry

def get_common_fastener(database, common_components, threshold=0.3):
    """
    获得常见的紧固件类型
    :param database: 数据集
    :param common_components: 常见零部件列表
    :param threshold: 阈值
    :return: 各零部件常见紧固件类型
    """
    component_fastener = {}
    for product in database.values():
        for item in product.values():
            if isinstance(item, dict):
                component_name = item['component_type'][0]
                fastener = item['fastener_type']
                component_fastener.setdefault(component_name, {})
                for fastener_type in fastener:
                    fastener_name = fastener_type[0]
                    if fastener_name not in component_fastener[component_name]:
                        component_fastener[component_name][fastener_name] = 1
                    else:
                        component_fastener[component_name][fastener_name] += 1
    # print(component_fastener, LOG='DEBUG')

    component_num, _ = get_component_num(database)

    common_fastener = {}
    for component, fastener in component_fastener.items():
        for fastener_name, fastener_num in fastener.items():
            if component in common_components and fastener_num / component_num[component] > threshold:
                common_fastener.setdefault(component, [])
                common_fastener[component].append(fastener_name)
    print('common_fastener:', common_fastener, LOG='INFO')

    return common_fastener

def get_common_fastener_features():
    # TODO: get common fastener features with 0.2 threshold
    pass

def get_common_component_colour(database, common_components, threshold=0.3):
    """
    获得常见的颜色
    :param database: 数据集
    :param common_components: 常见零部件列表
    :param threshold: 阈值
    :return: 各零部件常见颜色
    """
    component_colour = {}
    for product in database.values():
        for item in product.values():
            if isinstance(item, dict):
                component_name = item['component_type'][0]
                colour = item['colour'][0]
                component_colour.setdefault(component_name, {})
                if colour not in component_colour[component_name]:
                    component_colour[component_name][colour] = 1
                else:
                    component_colour[component_name][colour] += 1
    # print(component_colour, LOG='DEBUG')

    component_num, _ = get_component_num(database)

    common_colour = {}
    for component, colour in component_colour.items():
        for colour_name, colour_num in colour.items():
            if component in common_components and colour_num / component_num[component] > threshold:
                common_colour.setdefault(component, [])
                common_colour[component].append(colour_name)
    print('common_colour:', common_colour, LOG='INFO')

    return common_colour

def get_end_state_components(database, common_components):
    """
    获取拆卸最终状态的零部件列表
    :param database: 数据集
    :param common_components: 常见零部件列表
    :return: 拆卸最终状态的零部件列表
    """
    end_state_components = []
    for product in database.values():
        for item in product.values():
            if isinstance(item, dict):
                component_name = item['component_type'][0]
                if item['isEnd'][0] == True and component_name in common_components and component_name not in end_state_components:
                    end_state_components.append(item['component_type'][0])
    print('end_state_components:', end_state_components, LOG='INFO')

    return end_state_components


def get_rule(arg_dict, products, common_components, common_colour, component_common_geometry, smallest_larger_than,
             largest_smaller_than, immediately_preceding, end_state_components):
    """
    批量获得规则
    :param arg_dict:
    :param products:
    :param common_components:
    :param common_colour:
    :param component_common_geometry:
    :param smallest_larger_than:
    :param largest_smaller_than:
    :param immediately_preceding:
    :param end_state_components:
    :return:
    """
    whole_rule_dict = {}

    for component_name in common_components:
        for component_colour in common_colour[component_name]:
            [product_body, main_component_body, compare_size_body1, compare_size_body2, sub_component1_size_body,
             sub_component2_size_body, state_body, head] = ['', '', '', '', '', '', '', '']
            product_body = f"{arg_dict[products[0]]}(?p)"

            main_component_body = f"""{arg_dict['Component']}(?a0), {arg_dict[component_colour]}(?c0), 
                                    {arg_dict['hasColour']}(?a0, ?c0), {arg_dict['DisassemblyState']}(?a0, ?sta0), 
                                    {arg_dict[component_common_geometry[component_name]]}(?g0), {arg_dict['hasGeometry']}(?a0, ?g0), 
                                    {arg_dict['Size']}(?g0, ?size0), {arg_dict['isComponentOf']}(?a0, ?p)"""
            head = f"{arg_dict[component_name]}(?a0)"

            larger_size = smallest_larger_than[component_name][0]
            smaller_size = largest_smaller_than[component_name][0]
            if isinstance(larger_size, int):
                compare_size_body1 = f"greaterThan(?size0, {larger_size-1})"
                if isinstance(smaller_size, int):
                    compare_size_body2 = f"lessThan(?size0, {smaller_size+1})"
                else:
                    sub_component2 = smaller_size
                    sub_component2_size_body = f"""{arg_dict[sub_component2]}(?sc2), 
                                                                {arg_dict[component_common_geometry[sub_component2]]}(?sg2), {arg_dict['hasGeometry']}(?sc2, ?sg2), 
                                                                {arg_dict['Size']}(?sg2, ?sizesc2), {arg_dict['isComponentOf']}(?sc2, ?p), 
                                                                lessThan(?sizesc2, ?size0)"""
            elif isinstance(larger_size, str):
                sub_component1 = larger_size
                sub_component1_size_body = f"""{arg_dict[sub_component1]}(?sc1), 
                                    {arg_dict[component_common_geometry[sub_component1]]}(?sg1), {arg_dict['hasGeometry']}(?sc1, ?sg1), 
                                    {arg_dict['Size']}(?sg1, ?sizesc1), {arg_dict['isComponentOf']}(?sc1, ?p), 
                                    greaterThan(?sizesc1, ?size0)"""
                if isinstance(smaller_size, str):
                    sub_component2 = smaller_size
                    sub_component2_size_body = f"""{arg_dict[sub_component2]}(?sc2), 
                                            {arg_dict[component_common_geometry[sub_component2]]}(?sg2), {arg_dict['hasGeometry']}(?sc2, ?sg2), 
                                            {arg_dict['Size']}(?sg2, ?sizesc2), {arg_dict['isComponentOf']}(?sc2, ?p), 
                                            lessThan(?sizesc2, ?size0)"""

            preceding_component = immediately_preceding[component_name]
            if preceding_component != []:
                preceding_component = preceding_component[0]
                state_body = f"""{arg_dict[preceding_component]}(?stc), {arg_dict['DisassemblyState']}(?stc, ?sta1), 
                                            add(?sta2, ?sta1, 1),  equal(?sta2, ?sta0), 
                                            {arg_dict['isComponentOf']}(?stc, ?p)"""

            all_rule_list = [product_body, main_component_body, compare_size_body1, compare_size_body2,
                             sub_component1_size_body, sub_component2_size_body, state_body, head]
            all_body = ','.join([s.replace('\t', '').replace('\n', '').replace(' ', '') for s in all_rule_list[:-1] if s != ''])
            whole_rule = all_body + '->' + all_rule_list[-1]
            # print(whole_rule, LOG='DEBUG')
            whole_rule_dict[f'{products[0]}_{component_name}_with_{component_colour}'] = whole_rule

    if component_name in end_state_components:
        print('Disassembly to end state!', LOG='WARNING')
    else:
        raise ValueError('Disassembly wrong, no end state!')
    # print(whole_rule_list, LOG='DEBUG')
    return whole_rule_dict

def check_fastener(component_fasteners, common_fastener_list, name):
    """
    检查是否漏掉了某些零部件的紧固件
    :param component_fasteners: 检测到的零部件的紧固件
    :param common_fastener_list: 常见的紧固件列表
    :param name: (零部件所属类名, 零部件实例名)
    :return: 固定格式的紧固件字符串
    """
    # print(component_fasteners, LOG='DEBUG')
    # print(common_fastener_list, LOG='DEBUG')
    fastener_str = ''
    for common_fastener in common_fastener_list:
        if common_fastener not in component_fasteners.keys():
            print(f'[{name[0]}] component named [{name[1]}] may ignore the fastener named [{common_fastener}].', LOG='WARNING')

    for fastener_type, position in component_fasteners.items():
        if fastener_type != '':
            str = f'{fastener_type}, {position}, '
            fastener_str += str

    fastener_str = fastener_str[:-2] if len(fastener_str) > 2 else fastener_str
    # print(fastener_str, LOG='DEBUG')
    return fastener_str


def generate_data_in_format(products, onto, end_state_component_instance, common_fastener,
                            match2err_dict):
    """
    拆卸过程完成后，把本次拆卸涉及的数据转换为固定格式的数据
    :param products: 拆卸中的所有产品
    :param onto: 本体
    :param end_state_component_instance: 为最终拆卸状态的零部件实例
    :param common_fastener: 常见的零部件列表
    :return: 固定格式的数据
    """
    new_data = []
    err_dict = {}
    [err_dict.setdefault(i, 0) for i in range(0, 8)]
    for product, components in products.items():
        ID = product
        product_type = 'LCDmonitor'
        for component_name, component_features in components.items():
            disassembly_state = component_features[2]

            class_name = str(get_by_name(onto, component_name).is_a[-1])[4:]
            # print(class_name, LOG='DEBUG')
            if class_name == 'Component':
                err_dict[disassembly_state] += 1
                class_name = match2err_dict[disassembly_state]


            component_geometry = component_features[1]
            component_coordinates = '0-0'
            component_size = component_features[3]

            component_colour = component_features[0]

            component_fastener = check_fastener(component_features[4], common_fastener[class_name], [class_name, component_name])
            # component_fastener = ''

            if component_name in end_state_component_instance:
                isEnd = 'True'
            else:
                isEnd = 'False'

            component_str = f'{ID}, {product_type}; {disassembly_state}; {class_name}; {component_geometry}, ' \
                            f'{component_coordinates}, {component_size}; {component_colour}; {component_fastener}; ' \
                            f'{isEnd}'
            # print(component_str, LOG='DEBUG')
            new_data.append([component_str])
    err_dict = {k: err_dict[k] for k in sorted(err_dict.keys())}
    return new_data, err_dict

def pyrule_turn_to_SWRLrule(str_rule):
    """
    把python中的规则转换为Protégé中可识别的规则
    :param str_rule: python中的规则
    :return: Protégé中可识别的规则
    """
    return str_rule.replace('), ', ') ^ ')

if __name__ == '__main__':
    str_rule = 'LCDmonitor(?p), Component(?a0), BlackColour(?c0), hasColour(?a0, ?c0), DisassemblyState(?a0, ?sta0), Rectangle(?g0), hasGeometry(?a0, ?g0), Size(?g0, ?size0), isComponentOf(?a0, ?p), greaterThan(?size0, 3572), lessThan(?size0, 4334) -> Backcover(?a)'
    result = pyrule_turn_to_SWRLrule(str_rule)
    print(result, LOG='DEBUG')

