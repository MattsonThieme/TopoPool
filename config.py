
bs  = {'Caco2_Wang': 128,
       'PPBR_AZ':    128,
       'PROTEINS':   64,
       'ENZYMES':    128,
       'MUTAG':      32}

lrs = {'Caco2_Wang': 5e-3,
       'PPBR_AZ':    5e-3, 
       'PROTEINS':   1e-3,
       'ENZYMES':    1e-3,
       'MUTAG':      1e-3}

patiences ={'Caco2_Wang': 200,
            'PPBR_AZ':    200, 
            'PROTEINS':   200,
            'ENZYMES':    200,
            'MUTAG':      300}

evaluators = {'Caco2_Wang': ['MAE', 'lib'], 
              'PPBR_AZ':    ['MAE', 'lib'], 
              'PROTEINS':   ['CrossEntropy', 'lib'],
              'ENZYMES':    ['CrossEntropy', 'lib'],
              'MUTAG':      ['CrossEntropy', 'lib']}

