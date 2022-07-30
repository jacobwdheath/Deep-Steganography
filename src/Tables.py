import termtables as tt


class FormatTable:

    def __init__(self):
        self.variables = []

    def shapes(self, variables):
        self.variables = variables

        rows = []

        for i in range(len(self.variables)):
            size = self.variables[i].shape[0]
            x = self.variables[i].shape[1]
            y = self.variables[i].shape[2]
            colour_channels = self.variables[i].shape[3]

            if i != len(self.variables) - 1:
                rows.append(['Secret', size, x, y, colour_channels])
            else:
                rows.append(['Cover', size, x, y, colour_channels])

        table = tt.to_string(rows, header=['image type', 'size', 'width', 'height', 'channels'],
                             style=tt.styles.ascii_thin_double)
        return table
