from owlready2 import *

from data.database import ALL_DATA
from ontology_dataset import gen_eval_dataset, gen_dataset
from utilities import *
from tool import *
# tool.DEBUG = True

from logLEVEL import print
print.LEVEL = LEVEL = 'INFO'

def run(eval_products):
    data = get_from_csv('dataset.csv')
    # print(data, LOG='DEBUG')

    database = get_database(data, startLine=0)

    products = get_products(database)
    common_components = get_common_components(database)

    preceding_components, immediately_preceding = get_preceding_and_immediately_components(database, common_components)

    smallest_larger_than, largest_smaller_than = get_larger_and_smaller_than(database, common_components,
                                                                             preceding_components=preceding_components)

    component_common_geometry = get_component_common_geometry(database)

    common_fastener = get_common_fastener(database, common_components)

    common_colour = get_common_component_colour(database, common_components)

    end_state_components = get_end_state_components(database, common_components)

    onto = get_ontology("./LCD.owl")
    onto.load()
    print(onto, LOG='DEBUG')

    # print_all_classes_hierarchy(onto)
    # print_all_rules(onto)

    defined_classes(onto, 'Component', common_components)
    defined_classes(onto, 'Product', products)

    common_colour_list = list(set([c for colour_list in common_colour.values() for c in colour_list]))
    defined_classes(onto, 'Colour', common_colour_list)
    common_geometry_list = list(set(component_common_geometry.values()))
    defined_classes(onto, 'Geometry', common_geometry_list)

    # print_all_classes_hierarchy(onto)
    needed_args = {
        'class': [*common_colour_list, *common_geometry_list, *common_components, *products, 'Component'],
        'object_property': ['hasColour', 'hasGeometry', 'isComponentOf'],
        'data_property': ['Size', 'DisassemblyState']}
    arg_dict = get_iri(onto, needed_args)

    # print_all_classes_hierarchy(onto)
    # print(arg_dict, LOG='DEBUG')

    whole_rule_dict = get_rule(arg_dict, products, common_components, common_colour, component_common_geometry,
                               smallest_larger_than,
                               largest_smaller_than, immediately_preceding, end_state_components)

    with onto:
        # class aimAction(): pass
        for rule_name, infer_rule in whole_rule_dict.items():
            rule = Imp()
            rule.label = [rule_name]

            rule.set_as_rule(infer_rule)

    print_all_rules(onto)

    destroy_all_instance(onto)
    # print_instances(onto, name='Backcover')

    build_disassembly_product(onto, eval_products)

    sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True,
                         debug=1 if LEVEL == 'DEBUG' else 0)

    end_state_flag = False
    end_state_component_instance = []
    for end_state_component in end_state_components:
        end_state_component_instance.extend(list(get_by_name(onto, end_state_component).instances()))
        if end_state_component_instance != []:
            print(f'The product has been end state with [{end_state_component}]!', LOG='WARNING')
            end_state_flag = True

    end_state_component_instance = [str(item)[4:] for item in end_state_component_instance]

    match2err_dict = {
        0: 'Base', 1: 'Stand', 2: 'Back-cover', 3: 'PCB-cover', 4: 'Circuit-boards',
        5: 'Carrier', 6: 'LCD-module', 7: 'Front-bezel'
    }
    if end_state_component:
        new_data, err_dict = generate_data_in_format(eval_products, onto, end_state_component_instance, common_fastener,
                                                     match2err_dict)
        print(new_data, LOG='INFO')

        write_in_csv('new_data.csv', new_data)

        prod_num = len(eval_products)
        print(err_dict)
        acc = [(1 - err / prod_num) * 100 for err in err_dict.values()]
        print(f"ACC: {acc}")

if __name__ == '__main__':
    eval_num = 4

    DATA = [data for i, data_list in enumerate(ALL_DATA) if i != eval_num for data in data_list]
    gen_dataset(DATA)
    eval_products = gen_eval_dataset(ALL_DATA[eval_num])
    run(eval_products)

