import matplotlib.pyplot as plt

fields_to_use = ['Biology', 'Business', 'Sociology' , 'Computer science',
                 'Economics', 'Engineering', 'History', 'Mathematics',
                 'Medicine', 'Philosophy', 'Physics', 'Psychology', 'Chemistry']

overlap_field_order = [  
    'Economics',
    'Business',
    'Sociology',
    'Psychology',
    'Medicine',
    'Biology',
    'Chemistry',
    'Physics',
    'Engineering',
    'Mathematics',
    'Computer science',
    'History',
    'Philosophy', 
]

n_fields = len(fields_to_use)

# Assign consistent colors to fields
field_color_map = {field: plt.get_cmap('tab20')(i / n_fields) for i, field in enumerate(fields_to_use)}