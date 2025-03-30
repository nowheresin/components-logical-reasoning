import random

from tool import write_in_csv

def gen_dataset(DATA: list, filename='dataset'):
    dataset = []
    for i, data in enumerate(DATA):
        # gen_seq = ['Backcover', 'Carrier', 'PCBcover', 'PCB', 'LCDmodule', 'LCDmodule1']
        gen_seq = ['Base', 'Stand', 'Back-cover', 'PCB-cover', 'Circuit-boards', 'Carrier', 'LCD-module', 'Front-bezel']
        max_size = 10000
        temp_data = []
        for component in gen_seq:
            features = data[component]
            size = random.choice(features['Size'])
            # while size > max_size:
            #     size = random.choice(range(*features['Size']))
            # max_size = size
            content = f"P{i}, LCDmonitor; {random.choice(features['State'])}; {component}; {random.choice(features['Geometry']).capitalize()}, 0-0, {size}; {random.choice(features['Color']).replace(' ', '-')}Colour; ; "
            if component == 'Front-bezel':
                f = 'True'
            else:
                f = 'False'
            content += f

            temp_data.append([content])
        dataset.extend(temp_data)

    print(dataset)
    write_in_csv(f'{filename}.csv', dataset)

def gen_eval_dataset(DATA):
    # Function to convert component data into the desired format
    def _convert_to_product(component_data):
        color = component_data['Color'][0] + 'Colour'  # Extracting the color (assuming a single value)
        geometry = component_data['Geometry'][0].capitalize()  # Extracting the geometry (assuming a single value)
        size = int(component_data['Size'][0])  # Size as integer
        state = int(component_data['State'][0])  # State as integer
        return [color, geometry, state, size, {}]

    products = {}

    # Iterate over the input data to create different products (P1, P2, ...)
    for product_index, data in enumerate(DATA, start=1):
        product_id = f'P{product_index}'
        products[product_id] = {}

        # Add components to the product
        for component_index, (component_name, component_data) in enumerate(data.items(), start=1):
            component_id = f'C{component_index + (product_index - 1) * 8}'
            products[product_id][component_id] = _convert_to_product(component_data)
    print(products)
    return products

if __name__ == '__main__':
    from data.database import ALL_DATA
    DATA = [data for data_list in ALL_DATA for data in data_list]
    gen_dataset(DATA)
    gen_eval_dataset(ALL_DATA[2])
