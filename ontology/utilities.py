from owlready2 import *

from logLEVEL import print
print.LEVEL = 'DEBUG'

base_iri = 'http://www.LCD.com#'

def print_all_rules(onto):
    """
    打印本体中的所有swrl规则
    :param onto: 本体
    :return:
    """
    print(' SWRL rules '.center(60, '#'), LOG='INFO')
    # 获取本体文件中的所有SWRL规则
    for rule in onto.world.rules():
        if isinstance(rule, Imp):
            print(f'Rule named {rule.label}: {rule}', LOG='INFO')
    print(' end '.center(60, '#'), LOG='INFO')

def print_all_classes(onto):
    """
    打印本体中所有类，包括类名和IRI
    :param onto: 本体
    :return:
    """
    # 获取所有类及其IRI
    classes = list(onto.classes())
    # print(classes, LOG='DEBUG')
    # 打印所有类名及其IRI
    for class_obj in classes:
        print("类名: ", class_obj.name, LOG='INFO')
        print("IRI: ", class_obj.iri, LOG='INFO')
        print("".center(60, '-'), LOG='INFO')

def get_by_name(onto, name):
    """
    根据类的名称在本体中找到该类
    :param onto: 本体
    :param name: 类名
    :return: 类
    """
    return onto.search_one(iri=base_iri + name)

def print_name(onto, name='queryAction'):
    """
    根据类的名称在本体中找到该类，打印
    :param onto: 本体
    :param name: 类名
    :return:
    """
    class_name = get_by_name(onto, name)
    print(class_name, LOG='INFO')

def query_data_property_value(instance, data_property_name="DestructibilityDegree"):
    """
    查找某个实例中某个数据属性的值，打印
    :param instance: 实例
    :param data_property_name: 数据属性的名称
    :return: 该实例数据属性的值
    """
    # 查询数据属性值
    if hasattr(instance, data_property_name):
        data_property_value = getattr(instance, data_property_name)
        print("数据属性 {} 的值为: {}".format(data_property_name, data_property_value), LOG='INFO')

        return data_property_value

    raise ValueError(f'there is not data_property named [{data_property_name}] in [{instance.name}]')

def query_object_property_value(instance, object_property_name="usesTool"):
    """
    查找某个实例中某个对象属性的值，打印
    :param instance: 实例
    :param object_property_name: 对象属性的名称
    :return: 该实例对象属性的内容
    """
    # 查询对象属性值
    if hasattr(instance, object_property_name):
        object_properties = getattr(instance, object_property_name)
        for obj_prop in object_properties:
            print("对象属性 {} 的值为: {}".format(object_property_name, obj_prop.name), LOG='INFO')

        return object_properties

    raise ValueError(f'there is not object_property named [{object_property_name}] in [{instance.name}]')

def get_instance_by_name(onto, name):
    """
    根据实例名称找到实例
    :param onto: 本体
    :param name: 实例名称
    :return:
    """
    prop = get_by_name(onto, name)
    if prop is not None:
        return prop
    else:
        raise ValueError(f'there is no instance named [{name}] in [{onto.name}]')

def print_instances(onto, name='queryAction', sort=False, **kwargs):
    """
    打印某个类下的所有实例，可选打印该实例的单个对象属性/数据属性
    or 打印某个实例，可选打印该实例的单个对象属性/数据属性
    :param onto: 本体
    :param name: 类名/实例名
    :param sort: 是否把查询到的实例根据特定规则排序
    :param kwargs: 需要查询的对象属性/数据属性
    :return: 所有实例形成的字典
    """
    class_or_instance = get_by_name(onto, name)
    if isinstance(class_or_instance, ThingClass):
        type = 'class'
    elif isinstance(class_or_instance, Thing):
        type = 'instance'

    if type == 'class':
        # 查询该类的所有实例
        instances = list(class_or_instance.instances())
    elif type == 'instance':
        instances = [class_or_instance]

    Action = {}

    for instance in instances:
        print("实例名: ", instance.name, LOG='INFO')
        print("IRI: ", instance.iri, LOG='INFO')

        # Action[instance.name] = [instance.iri]
        Action[instance.name] = []

        if 'data_property_name' in kwargs:
            for item in kwargs['data_property_name']:
                data_property_value = query_data_property_value(instance, item)
                Action[instance.name].append(data_property_value[0])

        if 'object_property_name' in kwargs:
            for item in kwargs['object_property_name']:
                object_properties = query_object_property_value(instance, item)
                for obj_prop in object_properties:
                    Action[instance.name].append(obj_prop.name)

        # 如果实例还有其他属性，您也可以继续打印
        print("".center(60, '-'), LOG='INFO')

    if sort:
        Action = sorted(Action.items(), key=lambda d: d[1][0], reverse=False)
    if type == 'class':
        print(f'instances of class named [{name}]: ', dict(Action), LOG='INFO')
    elif type == 'instance':
        print(f'one instance named [{name}]: ', dict(Action), LOG='INFO')
    return Action

def print_all_classes_hierarchy(onto, Topping_name='Thing'):
    """
    按层次顺序打印某个顶层类下的所有子类
    :param onto: 本体
    :param Topping_name: 顶层类
    :return:
    """
    print(' classes hierarchy '.center(60, '#'), LOG='INFO')
    # 获取顶层类并开始打印
    if Topping_name == 'Thing':
        top_classes = [cls for cls in onto.classes() if owl.Thing in cls.is_a]
        print('Thing', LOG='INFO')
    else:
        aim_class = get_by_name(onto, Topping_name)
        top_classes = [cls for cls in onto.classes() if aim_class in cls.is_a]
        print(aim_class.name)

    # 打印本体中的类及其子类
    def _print_class_hierarchy(cls, level=1):
        print("|  " * (level - 1), end='', LOG='INFO')
        print("|--" + cls.name)
        # print(cls.is_a, LOG='DEBUG')
        for sub in cls.subclasses():
            _print_class_hierarchy(sub, level + 1)

    for cls in top_classes:
        _print_class_hierarchy(cls)
    print(' end '.center(60, '#'), LOG='INFO')

def print_all_object_properties(onto):
    """
    打印本体中所有对象属性
    :param onto: 本体
    :return:
    """
    for prop in onto.object_properties():
        print(prop, LOG='INFO')

def print_all_data_properties(onto):
    """
    打印本体中所有数据属性
    :param onto: 本体
    :return:
    """
    for prop in onto.data_properties():
        print(prop, LOG='INFO')

def get_object_property_by_name(onto, name):
    """
    根据名称在本体中获取对象属性
    :param onto: 本体
    :param name: 名称
    :return:
    """
    # for prop in onto.object_properties():
    #     if name in prop.iri:
    #         return prop
    prop = get_by_name(onto, name)
    if prop is not None:
        return prop
    else:
        raise ValueError(f'there is no object_property named [{name}] in [{onto.name}]')

def get_data_property_by_name(onto, name):
    """
    根据名称在本体中获取数据属性
    :param onto: 本体
    :param name: 名称
    :return:
    """
    prop = get_by_name(onto, name)
    if prop is not None:
        return prop
    else:
        raise ValueError(f'there is no data_properties named [{name}] in [{onto.name}]')

def get_iri(onto, needed_args):
    """
    接收一个字典，根据字典key，获取各参数的iri，返回一个字典
    :param onto: 本体
    :param needed_args: 参数字典
    :return: 参数及对应iri字典
    """
    arg_dict = {}
    for item in needed_args['class']:
        arg = get_by_name(onto, name=item).iri
        arg_dict[item] = arg
    for item in needed_args['object_property']:
        arg = get_object_property_by_name(onto, name=item).iri
        arg_dict[item] = arg
    for item in needed_args['data_property']:
        arg = get_data_property_by_name(onto, name=item).iri
        arg_dict[item] = arg

    return arg_dict

def replace_property(instance, property, aim):
    """
    修改某个实例某个对象属性/数据属性的值
    指向同一内存，所以不用返回！！
    :param instance: 实例
    :param property: 对象属性/数据属性名称
    :param aim: 目标值
    :return:
    """
    setattr(instance, property, [aim])

    # return instance


def add_object_property(instance, object_property, aim):
    """
    添加某个实例某个对象属性的值
    指向同一内存，所以不用返回！！
    :param instance: 实例
    :param object_property: 对象属性名称
    :param aim: 目标值
    :return:
    """
    # 获取属性的当前值
    current_value = getattr(instance, object_property)
    # 添加新的实例到属性中
    current_value.append(aim)

    # return instance

def get_instance(onto, item):
    """
    获取指定类的实例
    :param item: (类名, 实例名)
    :return: 实例
    """
    (class_name, instance_name) = item
    CLASS = get_by_name(onto, class_name)
    # print(CLASS, LOG='DEBUG')
    INSTANCE = CLASS(instance_name)
    INSTANCE.is_a.append(CLASS)
    # print_instances(onto, class_name)

    return INSTANCE

def build_disassembly_chain(onto, instance_dict):
    """
    建立fastener、features、detection的实例，以及和action的关系
    同一内存，不用返回！
    :param onto: 本体
    :param instance_dict: 各构件所需类以及实例的名称
    :return:
    """

    # must one fastener, one detection! (class_name, instance_name)
    fastener_instance = get_instance(onto, instance_dict['fastener'])
    feature_instances = [get_instance(onto, item) for item in instance_dict['features']]
    detection_instance = get_instance(onto, instance_dict['detection'])

    for instance in feature_instances:
        add_object_property(fastener_instance, 'hasFeature', instance)
        add_object_property(detection_instance, 'canDetect', instance)

    # 获取拆解行为，并添加拆解对象
    Action = get_instance_by_name(onto, instance_dict['Action'])
    add_object_property(Action, 'hasDestructibility', fastener_instance)

    # return onto

def build_disassembly_product(onto, products):
    for product, components in products.items():
        P1 = get_instance(onto, ('LCDmonitor', product))
        for component_name, component_features in components.items():
            colour, geometry, state, size, fastener = component_features
            C1 = get_instance(onto, ('Component', component_name))
            if not get_by_name(onto, colour):
                defined_classes(onto, 'Colour', [colour])
            C1_Colour = get_instance(onto, (colour, f'{component_name}_{colour}'))
            if not get_by_name(onto, geometry):
                defined_classes(onto, 'Geometry', [geometry])
            C1_Geometry = get_instance(onto, (geometry, f'{component_name}_{geometry}'))
            replace_property(C1, 'hasColour', C1_Colour)
            replace_property(C1, 'DisassemblyState', state)
            replace_property(C1, 'hasGeometry', C1_Geometry)
            replace_property(C1, 'isComponentOf', P1)
            replace_property(C1_Geometry, 'Size', size)


def defined_classes(onto, father_class, classes_list):
    father_class = get_by_name(onto, father_class)
    for sub_class in classes_list:
        type(sub_class, (father_class, ), {})

def destroy_all_instance(onto):
    # 销毁所有实例
    with onto:
        for entity in list(onto.individuals()):
            destroy_entity(entity)


if __name__ == '__main__':
    onto = get_ontology("./LCD-infer.owl")
    onto.load()

    # LCDIIPCBcoverAdhesive-object
    Adhesive_class = get_by_name(onto, 'Adhesive')
    print(Adhesive_class)
    LCDIIPCBcoverAdhesive = Adhesive_class('LCDIIPCBcoverAdhesive')
    LCDIIPCBcoverAdhesive.is_a.append(Adhesive_class)
    print_instances(onto, 'Adhesive')

    # Feature1
    Edge_class = get_by_name(onto, 'Edge')
    print(Edge_class)
    LCDIIPCBcoverAdhesiveGeometry = Edge_class('LCDIIPCBcoverAdhesiveGeometry')
    LCDIIPCBcoverAdhesiveGeometry.is_a.append(Edge_class)
    print_instances(onto, 'Edge')

    # Feature2
    SilverColour_class = get_by_name(onto, 'SilverColour')
    print(SilverColour_class)
    LCDIIPCBcoverAdhesiveColour = SilverColour_class('LCDIIPCBcoverAdhesiveColour')
    LCDIIPCBcoverAdhesiveColour.is_a.append(SilverColour_class)
    print_instances(onto, 'SilverColour')

    # Detection
    HoughLineDetection_class = get_by_name(onto, 'HoughLineDetection')
    print(HoughLineDetection_class)
    LCDIIPCBcoverAdhesiveDetection = HoughLineDetection_class('LCDIIPCBcoverAdhesiveDetection')
    LCDIIPCBcoverAdhesiveDetection.is_a.append(HoughLineDetection_class)
    print_instances(onto, 'HoughLineDetection')

    # 给LCDIIPCBcoverAdhesive添加特征
    add_object_property(LCDIIPCBcoverAdhesive, 'hasFeature', LCDIIPCBcoverAdhesiveGeometry)
    add_object_property(LCDIIPCBcoverAdhesive, 'hasFeature', LCDIIPCBcoverAdhesiveColour)
    query_item = {'object_property_name': ['hasFeature']}
    print_instances(onto, 'Adhesive', **query_item)

    # 给LCDIIPCBcoverAdhesiveDetection添加检测对象
    add_object_property(LCDIIPCBcoverAdhesiveDetection, 'canDetect', LCDIIPCBcoverAdhesiveGeometry)
    add_object_property(LCDIIPCBcoverAdhesiveDetection, 'canDetect', LCDIIPCBcoverAdhesiveColour)
    query_item = {'object_property_name': ['canDetect']}
    print_instances(onto, 'HoughLineDetection', **query_item)

    # 获取拆解行为，并添加拆解对象
    GenericDrillingAction = get_instance_by_name(onto, 'GenericDrillingAction')
    add_object_property(GenericDrillingAction, 'hasDestructibility', LCDIIPCBcoverAdhesive)
    query_item = {'object_property_name': ['hasDestructibility']}
    print_instances(onto, 'GenericDrillingAction', **query_item)

    # 定义要查询的数据属性和对象属性名
    query_item = {'data_property_name': ['DestructibilityDegree'], 'object_property_name': ['usesTool']}
    Action = print_instances(onto, name='queryAction', sort=True, **query_item)

    print_all_classes_hierarchy(onto, Topping_name='Action')
    # print_all_classes_hierarchy(onto)

    needed_args = {'class': ['DisassemblyAction', 'PerceptionAction', 'queryAction'],
                   'object_property': ['hasDestructibility', 'hasFeature', 'isDetectedBy', 'usesTool'],
                   'data_property': ['DestructibilityDegree']}

    arg_dict = get_iri(onto, needed_args)

    with onto:
        # class aimAction(): pass
        rule = Imp()
        rule.set_as_rule(
            f"""{arg_dict['DisassemblyAction']}(?a), {arg_dict['hasDestructibility']}(?a, ?o), {arg_dict['hasFeature']}(?o, ?f), {arg_dict['isDetectedBy']}(?f, ?d), {arg_dict['PerceptionAction']}(?d) -> {arg_dict['queryAction']}(?a)""")

    sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)

    Action = print_instances(onto, name='queryAction', sort=True, **query_item)

