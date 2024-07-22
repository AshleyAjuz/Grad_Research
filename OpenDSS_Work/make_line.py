
def make_line(dss, load):

    dss.loads_write_name(load.lower().split('.')[1])

    load_bus = dss.cktelement_read_bus_names()

    #find where that load bus exists as Bus 1 in the lines

    names_lines = dss.lines_all_names()

    dss.lines_first()

    for line in names_lines:

        if load_bus[0][:3] == dss.lines_read_bus1()[:3]:

            load_line = line

            load_line_phase = dss.lines_read_phases()

            load_line_bus = dss.lines_read_bus1()

            load_line_buscode = dss.lines_read_linecode()

        elif load_bus[0][:3] == dss.lines_read_bus2()[:3]:

            load_line = line

            load_line_phase = dss.lines_read_phases()

            load_line_bus = dss.lines_read_bus2()

            load_line_buscode = dss.lines_read_linecode()

        dss.lines_next()

 

    if '_new' in load_line:

    #if load_line in names_lines:

        new_load_line_bus = load_line_bus[:3]+'_new' + load_bus[0][3:]

        dss.text(f'edit {load} Bus1={new_load_line_bus}')

    elif load_line +'_new' in names_lines:

    #if load_line in names_lines:

        new_load_line_bus = load_line_bus[:3]+'_new' + load_bus[0][3:]

        dss.text(f'edit {load} Bus1={new_load_line_bus}')

    else:

        dss.text(f'New Line.{load_line}_new Phases={load_line_phase} Bus1={load_line_bus[:3]}.1.2.3 Bus2={load_line_bus[:3]}_new.1.2.3 LineCode={load_line_buscode}  Length=0.01')

        new_load_line_bus = load_line_bus[:3]+'_new' + load_bus[0][3:]

        dss.text(f'edit {load} Bus1={new_load_line_bus}')

 

    return dss