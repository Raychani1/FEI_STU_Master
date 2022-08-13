import os

from text_corrector import TextCorrector

if __name__ == '__main__':

    text_corrector = TextCorrector(mode='custom')

    text_corrector.run_correction_pipeline()

    # Presentation
    
    # If custom Dictionary is given change value [text_corrector.py, line 51]

    # text_corrector = TextCorrector(
    #     mode='custom',
    #     compare_directory=os.path.join(
    #         os.getcwd(),
    #         'data',
    #         'presentation',
    #         'compare'
    #     ),
    #     input_directory=os.path.join(
    #         os.getcwd(),
    #         'data',
    #         'presentation',
    #         'input'
    #     ),        
    #     output_directory=os.path.join(
    #         os.getcwd(),
    #         'data',
    #         'presentation',
    #         'output'
    #     )
    # )

    # text_corrector.run_correction_pipeline()
