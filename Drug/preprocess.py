class Example():
    def __init__(self, compound_id, assay_id, pic50_exp, pic50_pred):
        self.compound_id = compound_id
        self.assay_id = assay_id
        self.pic50_exp = pic50_exp
        self.pic50_pred = pic50_pred


class Experiment():
    def __init__(self, training_set, test_set, assays, compounds, mode):
        self.training_set = training_set
        self.test_set = test_set
        self.assays = assays
        self.compounds = compounds
        self.mode = mode  # random or realistic


def read_example(line, compounds):
    strings = line.split(',')
    compound_id = str(strings[0])
    pic50_exp = float(strings[1])
    pic50_pred = float(strings[2])
    assay_id = int(strings[3])
    smiles = strings[4]

    if compound_id not in compounds:
        compounds[compound_id] = smiles

    example = Example(compound_id, assay_id, pic50_exp, pic50_pred)

    return example


def read_4276_txt(filename, compound_filename):
    # first of all, read all the compounds
    compound_file = open(compound_filename, 'r', encoding='UTF-8', errors='ignore')
    clines = compound_file.readlines()
    compound_file.close()

    compounds = {}
    previous = ''
    previous_id = ''
    for cline in clines:
        cline = str(cline.strip())
        if 'CHEMBL' not in cline:
            if 'Page' in cline or cline == '' or 'Table' in cline or 'SMILE' in cline:
                continue
            else:
                previous += cline
        else:
            strings = cline.split(',')

            if previous_id not in compounds and previous != '':
                compounds[previous_id] = previous.replace('\u2010', '-')

            previous_id = strings[0]
            previous = strings[1]

    compounds[previous_id] = previous.replace('\u2010', '-')

    assay_ids = []
    real_train = {}
    real_test = {}

    file = open(filename, 'r', encoding='UTF-8', errors='ignore')
    lines = file.readlines()
    file.close()

    for line in lines:
        line = str(line.strip())
        if 'CHEMBL' not in line:
            continue
        strings = line.split(' ')
        compound_id = str(strings[0])
        assay_id = int(strings[1])
        try:
            pic50_exp = float(strings[2])
        except:
            pic50_exp = -float(strings[2][1:])
        try:
            pic50_pred = float(strings[3])
        except:
            pic50_pred = -float(strings[3][1:])
        train_flag = strings[4]

        if assay_id not in assay_ids:
            assay_ids.append(assay_id)

        tmp_example = Example(compound_id, assay_id, pic50_exp, pic50_pred)

        if train_flag == 'TRN':
            if assay_id not in real_train:
                real_train[assay_id] = []
                real_train[assay_id].append(tmp_example)
            else:
                real_train[assay_id].append(tmp_example)
        else:
            if assay_id not in real_test:
                real_test[assay_id] = []
                real_test[assay_id].append(tmp_example)
            else:
                real_test[assay_id].append(tmp_example)

    experiment = Experiment(real_train, real_test, assay_ids, compounds, 'realistic')

    return experiment