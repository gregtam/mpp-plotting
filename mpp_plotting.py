from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from textwrap import dedent

from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import psycopg2
import seaborn as sns
import sqlalchemy
from sqlalchemy import create_engine, Column, MetaData, Table
from sqlalchemy import all_, and_, any_, not_, or_
from sqlalchemy import alias, between, case, cast, column, false, func,\
                       intersect, literal, literal_column, select, text, true
from sqlalchemy import BigInteger, Boolean, Date, DateTime, Integer, Float,\
                       Numeric, String

import credentials


def _add_weights_column(df_list, normed):
    """Add the weights column for each DataFrame in a list of
    DataFrames.
    """

    for df in df_list:
        df['weights'] = _create_weight_percentage(df[['freq']], normed)


def _create_weight_percentage(hist_col, normed=False):
    """Convert frequencies to percent."""
    if normed:
        return hist_col/hist_col.sum()
    else:
        return hist_col


def _listify(df_list, labels):
    """If df_list and labels are DataFrames and strings respectively,
    make them into lists to conform with the rest of the code as it is
    built to handle multiple variables.
    """

    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]
    if isinstance(labels, str):
        labels = [labels]
    return df_list, labels


def _separate_schema_table(full_table_name, con):
    """Separates schema name and table name.
    
    Inputs:
    full_table_name - Schema and table name together joined by a '.'
    con - A SQLAlchemy engine or psycopg2 connection object
    """
    if '.' in full_table_name:
        return full_table_name.split('.')
    else:
        if isinstance(con, psycopg2.extensions.connection):
            schema_name = psql.read_sql('SELECT current_schema();', con).iloc[0, 0]
        elif isinstance(con, sqlalchemy.engine.base.Engine):
            schema_name = con.execute(text('SELECT current_schema();')).scalar()
        table_name = full_table_name
        return schema_name, full_table_name



def get_histogram_values(table_obj, column_name, con, metadata, nbins=25,
                         bin_width=None, cast_as=None, print_query=False):
    """Takes a SQL table and creates histogram bin heights. Relevant
    parameters are either the number of bins or the width of each bin.
    Only one of these is specified. The other one must be left at its
    default value of 0 or it will throw an error.
    
    Inputs:
    table_name - Name of the table in SQL. Input can also include have
                 the schema name prepended, with a '.', e.g.,
                 'schema_name.table_name'
    column_name - Name of the column of interest
    con - A SQLAlchemy engine or psycopg2 connection object
    nbins - Number of desired bins (Default: 25)
    bin_width - Width of each bin (Default: None)
    cast_as - SQL type to cast as (string or SQLAlchemy data type)
    where_clause - A SQL where clause specifying any filters
    print_query - If True, print the resulting query.
    """

    def _check_for_input_errors(nbins, bin_width):
        """Check to see if any inputs conflict and raise an error if
        there are issues.
        """

        if nbins is not None and nbins < 0:
            raise Exception('nbins must be positive.')
        if bin_width is not None and bin_width < 0:
            raise Exception('bin_width must be positive.')
    
    def _get_column_information(full_table_name, column_name, con):
        """Get column name and data type information."""
        schema_name, table_name =  _separate_schema_table(full_table_name, con)

        sql = '''
        SELECT column_name, data_type
          FROM information_schema.columns
         WHERE table_schema = '{schema_name}'
           AND table_name = '{table_name}'
           AND column_name = '{column_name}'
        '''.format(**locals())
        
        return psql.read_sql(sql, con)
    
    def _is_category_column(table_obj, column_name):
        """Returns whether the column is a category."""

        data_type = str(table_obj.c[column_name].type)
        numeric_types = ['DATE', 'DOUBLE PRECISION', 'INT', 'FLOAT',
                         'NUMERIC', 'TIMESTAMP',
                         'TIMESTAMP WITHOUT TIME ZONE']
        return data_type not in numeric_types

    def _is_time_type(table_obj, column_name):
        """Returns whether the column is a time type (date or timestamp)."""
        data_type = str(table_obj.c[column_name].type)
        time_types = ['DATE', 'TIMESTAMP', 'TIMESTAMP WITHOUT TIME ZONE']
        return data_type in time_types

    def _get_cast_string(cast_as):
        """If cast_as is specified, we must create a cast string to
        recast our columns. If not, we set it as a blank string.
        """

        if cast_as is None:
            return ''
        else:
            return '::' + cast_as.upper()

    def _min_max_value(table_obj, column_name, con, cast_as=None):
        desired_col = column(column_name)
        if cast_as is not None:
            desired_col = desired_col.cast(cast_as)

        min_max_sql =\
            select([func.min(desired_col), func.max(desired_col)],
                   from_obj=table_obj
                  )

        return tuple(psql.read_sql(min_max_sql, con).iloc[0])
    
    _check_for_input_errors(nbins, bin_width)
    # info_df = _get_column_information(table_name, column_name, con)
    is_category = _is_category_column(table_obj, column_name)
    is_time_type = _is_time_type(table_obj, column_name)
    # cast_string = _get_cast_string(cast_as)

    if is_category:
        sql =\
            select([column(column_name).label('category'),
                    func.count('*').label('freq')
                   ],
                   from_obj=table_obj
                  )\
            .group_by(column_name)\
            .order_by(column('freq').desc())
    elif is_time_type:
        # Get min and max value of the column
        min_val, max_val = _min_max_value(table_obj, column_name, con)
        col_val = column(column_name)
        min_time = literal(min_val)
        max_time = literal(max_val)
        
        # Get the span of the column
        span_value = max_val - min_val
        if bin_width is not None:
            # If bin width is specified, calculate nbins from it.
            nbins = span_value/bin_width

        print span_value, type(span_value)

        # Get the SQL expressions for the time ranges
        time_pos_numer = func.extract('EPOCH', col_val - max_time)
        time_range_denom = func.extract('EPOCH', min_time - max_time)
        # Which bin it should fall into
        bin_nbr = func.floor(time_pos_numer/time_range_denom * nbins)
        # Scale the bins to their proper size
        bin_nbr_scaled = bin_nbr/nbins * time_range_denom
        # Translate bins to their proper locations
        bin_loc = bin_nbr_scaled * text("INTERVAL '1 second'") + min_time

        binned_table =\
            select([bin_loc.label('bin_loc'),
                    func.count('*').label('freq')
                   ], 
                   from_obj=table_obj
                  )\
            .group_by('bin_loc')\
            .order_by('bin_loc')

        return psql.read_sql(binned_table, con)
        
    else:
        # Set column variables
        min_val = column('min_val')
        max_val = column('max_val')
        col_val = column(column_name)
        
        # Get the span of the column
        span_value = max_val - min_val
        if bin_width is not None:
            # If bin width is specified, calculate nbins from it.
            nbins = span_value/bin_width

        # Which bin it should fall into
        bin_nbr =  func.floor((col_val - min_val)/span_value * nbins)
        # Scale the bins to their proper size
        bin_nbr_scaled = bin_nbr/nbins * span_value
        # Translate bins to their proper locations
        bin_loc = bin_nbr_scaled + min_val

        min_max_tbl =\
            select([func.min(column('x')).label('min_val'),
                    func.max(column('y')).label('max_val')
                   ],
                   from_obj=table_obj
                   )\
            .alias('foo')

        binned_table =\
            select([bin_loc.label('bin_loc'),
                    func.count('*').label('freq')
                   ], 
                   from_obj=[table_obj, min_max_tbl]
                  )\
            .group_by('bin_loc')\
            .order_by('bin_loc')


        return psql.read_sql(binned_table, con)

    if print_query:
        print dedent(sql)

    return psql.read_sql(sql, con)


def get_roc_auc_score(roc_df, tpr_column='tpr', fpr_column='fpr'):
    """Given an ROC DataFrame such as the one created in get_roc_values,
    return the AUC. This is achieved by taking the ROC curve and 
    interpolating every single point with a straight line and computing
    the sum of the areas of all the trapezoids.

    Inputs:
    roc_df - A DataFrame with columns for true positive rate and false
             positive rate
    tpr_column - Name of the true positive rate column (Default: 'tpr')
    fpr_column - Name of the false positive rate column (Default: 'fpr')
    """

    # The average of the two consecutive tprs
    avg_height = roc_df[tpr_column].rolling(2).mean()[1:]
    # The width (i.e., distance between two consecutive fprs)
    width = roc_df[fpr_column].diff()[1:]

    return sum(avg_height * width)


def get_roc_values(table_name, y_true, y_score, conn, print_query=False):
    """Computes the ROC curve in database.

    Inputs:
    table_name - The name of the table that includes predicted and true
                 values
    y_true - The name of the column that contains the true values
    y_score - The name of the column that contains the scores of the
              machine learning algorithm
    conn - A psycopg2 connection object
    print_query - If True, print the resulting query.
    """

    sql = '''
      WITH row_num_table AS
           (SELECT row_number()
                       OVER (ORDER BY {y_score}) AS row_num,
                   *
              FROM {table_name}
           ),
           pre_roc AS 
           (SELECT *,
                   SUM({y_true})
                       OVER (ORDER BY {y_score} DESC) AS num_pos,
                   SUM(1 - {y_true})
                       OVER (ORDER BY {y_score} DESC) AS num_neg
              FROM row_num_table
           ),
           class_sizes AS
           (SELECT SUM({y_true}) AS tot_pos,
                   SUM(1 - {y_true}) AS tot_neg
              FROM {table_name}
           )
    SELECT DISTINCT
           {y_score} AS thresholds,
           num_pos/tot_pos::NUMERIC AS tpr,
           num_neg/tot_neg::NUMERIC AS fpr
      FROM pre_roc
           CROSS JOIN class_sizes
     ORDER BY tpr, fpr;
    '''.format(**locals())

    if print_query:
        print dedent(sql)

    return psql.read_sql(sql, conn)


def get_scatterplot_values(table_name, column_name_x, column_name_y, conn,
                           nbins=(1000, 1000), bin_size=None, cast_x_as=None,
                           cast_y_as=None, print_query=False):
    """ Takes a SQL table and creates scatter plot bin values. This is
    the 2D version of get_histogram_values. Relevant parameters are
    either the number of bins or the size of each bin in both the x and
    y direction. Only number of bins or size of the bins is specified.
    The other pair must be left at its default value of 0 or it will
    throw an error.
    
    Inputs:
    table_name - Name of the table in SQL. Input can also include have
                 the schema name prepended, with a '.', e.g.,
                 'schema_name.table_name'
    column_name_x - Name of one column of interest to be plotted
    column_name_y - Name of another column of interest to be plotted
    conn - A psycopg2 connection object
    column_name - Name of the column of interest
    nbins - Number of desired bins for x and y directions
            (Default: (0, 0))
    bin_size - Size of each bin for x and y directions (Default: (0, 0))
    cast_x_as - SQL type to cast x as
    cast_y_as - SQL type to cast y as
    print_query - If True, print the resulting query.
    """

    def _check_for_input_errors(nbins, bin_size):
        """Check to see if any inputs conflict and raise an error if
        there are issues.
        """

        if bin_size is not None:
            if bin_size[0] < 0 or bin_size[1] < 0:
                raise Exception('Bin dimensions must both be positive.')
        elif nbins is not None:
            if nbins[0] < 0 or nbins[1] < 0:
                raise Exception('Number of bin dimensions must both be positive')
    
    def _get_cast_string(cast_as_x, cast_as_y):
        """If cast_as_x and/or cast_as_y are specified, we must create a
        cast string to recast our columns. If not, we set it as a blank
        string.
        """

        if cast_x_as is None:
            cast_x_string = ''
        else:
            cast_x_string = '::' + cast_x_as.upper()
            
        if cast_y_as is None:
            cast_y_string = ''
        else:
            cast_y_string = '::' + cast_y_as.upper()

        return cast_x_string, cast_y_string

    def _min_max_value(table_name, column_name, cast_as):
        """Get the min and max value of a specified column."""
        sql = '''
        SELECT MIN({column_name}{cast_as}), MAX({column_name}{cast_as})
          FROM {table_name};
        '''.format(**locals())

        return tuple(psql.read_sql(sql, conn).iloc[0])
     

    schema_name, table_name = _separate_schema_table(table_name, conn)
    _check_for_input_errors(nbins, bin_size)
    cast_x_string, cast_y_string = _get_cast_string(cast_x_as, cast_y_as)

    # Get the min and max values for x and y directions
    min_val_x, max_val_x = _min_max_value(table_name,
                                          column_name_x,
                                          cast_as=cast_x_string
                                         )
    min_val_y, max_val_y = _min_max_value(table_name,
                                          column_name_y,
                                          cast_as=cast_y_string
                                         )
    
    # Get the span of values in the x and y direction
    span_values = (max_val_x - min_val_x, max_val_y - min_val_y)
    
    # Since the bins are generated using nbins, if only bin_size is
    # specified, we can back calculate the number of bins that will be
    # used.
    if bin_size is not None:
        nbins = [float(i)/j for i, j in zip(span_values, bin_size)]

    sql = '''
    DROP TABLE IF EXISTS binned_table_temp;
    CREATE TABLE binned_table_temp
       AS SELECT FLOOR(({x_col}{cast_x_as} - {min_val_x})
                             /({max_val_x} - {min_val_x}) 
                             * {nbins_x}
                        )
                       /{nbins_x} * ({max_val_x} - {min_val_x}) 
                       + {min_val_x} AS bin_nbr_x,
                   FLOOR(({y_col}{cast_y_as} - {min_val_y})
                             /({max_val_y} - {min_val_y}) 
                             * {nbins_y}
                        )
                       /{nbins_y} * ({max_val_y} - {min_val_y}) 
                       + {min_val_y} AS bin_nbr_y
              FROM {table_name}
             WHERE {x_col} IS NOT NULL
               AND {y_col} IS NOT NULL;

    DROP TABLE IF EXISTS scatter_bins_temp;
    CREATE TABLE scatter_bins_temp
       AS SELECT *
              FROM (SELECT x::NUMERIC/{nbins_x} * ({max_val_x} - {min_val_x})
                               + {min_val_x} AS scat_bin_x
                     FROM generate_series(1, {nbins_x}) AS x
                   ) AS foo_x
                   CROSS JOIN (SELECT y::NUMERIC/{nbins_y} * ({max_val_y} - {min_val_y})
                                          + {min_val_y} AS scat_bin_y
                                 FROM generate_series(1, {nbins_y}) AS y 
                              ) AS foo_y;

      WITH binned_table AS
           (SELECT FLOOR(({x_col}{cast_x_as} - {min_val_x})
                             /({max_val_x} - {min_val_x}) 
                             * {nbins_x}
                        )
                       /{nbins_x} * ({max_val_x} - {min_val_x}) 
                       + {min_val_x} AS bin_nbr_x,
                   FLOOR(({y_col}{cast_y_as} - {min_val_y})
                             /({max_val_y} - {min_val_y}) 
                             * {nbins_y}
                        )
                       /{nbins_y} * ({max_val_y} - {min_val_y}) 
                       + {min_val_y} AS bin_nbr_y
              FROM {table_name}
             WHERE {x_col} IS NOT NULL
               AND {y_col} IS NOT NULL
           ),
           scatter_bins AS
           (SELECT *
              FROM (SELECT x::NUMERIC/{nbins_x} * ({max_val_x} - {min_val_x})
                               + {min_val_x} AS scat_bin_x
                     FROM generate_series(0, {nbins_x}) AS x
                   ) AS foo_x
                   CROSS JOIN (SELECT y::NUMERIC/{nbins_y} * ({max_val_y} - {min_val_y})
                                          + {min_val_y} AS scat_bin_y
                                 FROM generate_series(0, {nbins_y}) AS y 
                              ) AS foo_y
           )
    SELECT scat_bin_x, scat_bin_y, COUNT(bin_nbr_x) AS freq
      FROM binned_table
           RIGHT JOIN scatter_bins
                   ON ROUND(bin_nbr_x::NUMERIC, 6) = ROUND(scat_bin_x::NUMERIC, 6)
                  AND ROUND(bin_nbr_y::NUMERIC, 6) = ROUND(scat_bin_y::NUMERIC, 6)
     GROUP BY scat_bin_x, scat_bin_y
     ORDER BY scat_bin_x, scat_bin_y;
    '''.format(x_col = column_name_x,
               cast_x_as = cast_x_string,
               y_col = column_name_y,
               cast_y_as = cast_y_string,
               min_val_x = min_val_x - 1e-8,
               max_val_x = max_val_x + 1e-8,
               min_val_y = min_val_y - 1e-8,
               max_val_y = max_val_y + 1e-8,
               nbins_x = nbins[0],
               nbins_y = nbins[1],
               table_name = table_name
              )
    
    if print_query:
        print dedent(sql)

    return psql.read_sql(sql, conn)


def plot_categorical_hists(df_list, labels=[], log=False, normed=False,
                           null_at='left', order_by=0, ascending=True,
                           color_palette=sns.color_palette('deep')):
    """Plots categorical histograms.
    
    Inputs:
    df_list - A pandas DataFrame or a list of DataFrames which have two
              columns (bin_nbr and freq). The bin_nbr is the value of 
              the histogram bin and the frequency is how many values 
              fall in that bin.
    labels - A string (for one histogram) or list of strings which sets 
             the labels for the histograms
    log - Boolean of whether to display y axis on log scale
          (Default: False)
    normed - Boolean of whether to normalize histograms so that the 
             heights of each bin sum up to 1. This is useful for 
             plotting columns with difference sizes (Default: False)
    null_at - Which side to set a null value column. The options are:
              'left' - Put the null on the left side
              'order' - Leave it in its respective order
              'right' - Put it on the right side
              '' - If left blank, leave out              
              (Default: order)
    order_by - How to order the bars. The options are:
               'alphetical' - Orders the categories in alphabetical
                            order
               integer - an integer value denoting for which df_list
                         DataFrame to sort by
    ascending - Boolean of whether to sort values in ascending order 
                (Default: False)
    color_palette - Seaborn colour palette, i.e., a list of tuples
                    representing the colours. 
                    (Default: sns deep color palette)
    """

    def _join_freq_df(df_list):
        """Joins all the DataFrames so that we have a master table with
        category and the frequencies for each table.

        Returns the joined DataFrame
        """

        for i in range(len(df_list)):
            temp_df = df_list[i].copy()
            temp_df.columns = ['category', 'freq_{}'.format(i)]

            # Add weights column (If normed, we must take this into account)
            temp_df['weights_{}'.format(i)] = _create_weight_percentage(temp_df['freq_{}'.format(i)], normed)

            if i == 0:
                df = temp_df
            else:
                df = pd.merge(df, temp_df, how='outer', on='category')

        # Fill in nulls with 0 (except for category column)
        for col in df.columns[1:]:
            df[col] = df[col].fillna(0)
        return df
  
    def _get_num_categories(hist_df):
        """Get the number of categories depending on whether we are 
        specifying to drop it in the function.
        """

        if null_at == '':
            return hist_df['category'].dropna().shape[0]
        else:
            return hist_df.shape[0]
   
    def _get_bin_order(loc, hist_df, order_by):
        """Sorts hist_df by the specified order."""
        if order_by == 'alphabetical':
            return hist_df\
                .sort_values('category', ascending=ascending)\
                .reset_index(drop=True)
        elif str(type(order_by)) == "<type 'int'>":
            # Desired column in the hist_df DataFrame
            weights_col = 'weights_{}'.format(order_by)

            if weights_col not in hist_df.columns:
                raise Exception('order_by index not in hist_df.')
            return hist_df\
                .sort_values(weights_col, ascending=ascending)\
                .reset_index(drop=True)
        else:
            raise Exception('Invalid order_by')

    def _get_bin_left(loc, hist_df):
        """Returns a list of the locations of the left edges of the
        bins.
        """
        
        def _get_within_bin_left(hist_df):
            """Each bin has width 1. If there is more than one
            histogram, each one must fit in this bin of width 1, so

            Returns indices within a bin for each histogram.
            """

            if len(df_list) == 1:
                return [0, 1]
            else:
                return np.linspace(0.1, 0.9, num_hists + 1)[:-1]

        within_bin_left = _get_within_bin_left(hist_df)

        # For each histogram, we generate a separate list of tick
        # locations. We do this so that later, when we plot we can use
        # different colours for each.

        # If there are any NULL categories
        if np.sum(hist_df.category.isnull()) > 0:
            if loc == 'left': 
                bin_left = [np.arange(1 + within_bin_left[i], num_categories + within_bin_left[i]).tolist() for i in range(num_hists)]
                null_left = [[within_bin_left[i]] for i in range(num_hists)]
            elif loc == 'right':
                bin_left = [np.arange(within_bin_left[i], num_categories - 1 + within_bin_left[i]).tolist() for i in range(num_hists)]
                # Subtract one from num_categories since num_categories
                # includes the null bin. Subtracting will place the null 
                # bin in the proper location.
                null_left = [[num_categories - 1 + within_bin_left[i]] for i in range(num_hists)]
            elif loc == 'order':
                # Get the index of null and non-null categories in
                # hist_df
                null_indices = np.array(hist_df[pd.isnull(hist_df.category)].index)
                non_null_indices = np.array(hist_df.dropna().index)
                bin_left = [(within_bin_left[i] + non_null_indices).tolist() for i in range(num_hists)]
                null_left = [(within_bin_left[i] + null_indices).tolist() for i in range(num_hists)]
            elif loc == '':
                bin_left = [np.arange(within_bin_left[i], num_categories + 1 + within_bin_left[i])[:-1].tolist() for i in range(num_hists)]
                null_left = [[]] * num_hists
        else:
            bin_left = [np.arange(within_bin_left[i], hist_df.dropna().shape[0] + 1 + within_bin_left[i])[:-1].tolist() for i in range(num_hists)]
            null_left = [[]] * num_hists

        return bin_left, null_left

    def _get_bin_height(loc, order_by, hist_df):
        """Returns a list of the heights of the bins and the category
        order.
        """

        hist_df_null = hist_df[hist_df.category.isnull()]
        hist_df_non_null = hist_df[~hist_df.category.isnull()]

        # Set the ordering
        if order_by == 'alphabetical':            
            hist_df_non_null = hist_df_non_null\
                .sort_values('category', ascending=ascending)
        else:
            if 'weights_{}'.format(order_by) not in hist_df_non_null.columns:
                raise Exception('Order by number exceeds number of DataFrames.')
            hist_df_non_null = hist_df_non_null\
                .sort_values('weights_{}'.format(order_by), ascending=ascending)

        if log:
            bin_height = [np.log10(hist_df_non_null['weights_{}'.format(i)]).tolist() for i in range(num_hists)]
        else:
            bin_height = [hist_df_non_null['weights_{}'.format(i)].tolist() for i in range(num_hists)]

        # If loc is '', then we do not want a NULL height
        # since we are ignoring NULL values
        if loc == '':
            null_height = [[]] * num_hists
        else:
            if log:
                null_height = [np.log10(hist_df_null['weights_{}'.format(i)]).tolist() for i in range(num_hists)]
            else:
                null_height = [hist_df_null['weights_{}'.format(i)].tolist() for i in range(num_hists)]

        return bin_height, null_height

    def _get_bin_width(num_hists):
        """Returns each bin width based on number of histograms."""
        if num_hists == 1:
            return 1
        else:
            return 0.8/num_hists

    def _plot_all_histograms(bin_left, bin_height, null_bin_left,
                             null_bin_height, bin_width):
        for i in range(num_hists):
            # If there are any null bins, plot them
            if len(null_bin_height[i]) > 0:
                plt.bar(null_bin_left[i], null_bin_height[i], bin_width,
                        hatch='x', color=color_palette[i])
            plt.bar(bin_left[i], bin_height[i], bin_width,
                    color=color_palette[i])

    def _plot_xticks(loc, bin_left, hist_df):
        # If there are any NULL categories
        if np.sum(hist_df.category.isnull()) > 0:
            if loc == 'left':
                xticks_loc = np.arange(num_categories) + 0.5
                plt.xticks(xticks_loc,
                           ['NULL'] + hist_df.dropna()['category'].tolist(),
                           rotation=90
                          )
            elif loc == 'right':
                xticks_loc = np.arange(num_categories) + 0.5
                plt.xticks(xticks_loc,
                           hist_df.dropna()['category'].tolist() + ['NULL'],
                           rotation=90
                          )
            elif loc == 'order':
                xticks_loc = np.arange(num_categories) + 0.5
                plt.xticks(xticks_loc,
                           hist_df['category'].fillna('NULL').tolist(),
                           rotation=90
                          )
            elif loc == '':
                xticks_loc = np.arange(num_categories) + 0.5
                plt.xticks(xticks_loc,
                           hist_df.dropna()['category'].tolist(),
                           rotation=90
                          )
        else:
            xticks_loc = np.arange(num_categories) + 0.5
            plt.xticks(xticks_loc,
                       hist_df.dropna()['category'].tolist(),
                       rotation=90
                      )

    def _plot_new_yticks(bin_height):
        """Changes yticks to log scale."""
        max_y_tick = int(np.ceil(np.max(bin_height))) + 1
        yticks = [10**i for i in range(max_y_tick)]
        yticks = ['1e{}'.format(i) for i in range(max_y_tick)]
        plt.yticks(range(max_y_tick), yticks)


    df_list, labels = _listify(df_list, labels)
    # Joins in all the df_list DataFrames so that we can pick a certain 
    # category and retrieve the count for each.
    hist_df = _join_freq_df(df_list)
    # Order them based on specified order
    hist_df = _get_bin_order(null_at, hist_df, order_by)

    num_hists = len(df_list)
    num_categories = _get_num_categories(hist_df)

    bin_left, null_bin_left = _get_bin_left(null_at, hist_df)
    bin_height, null_bin_height = _get_bin_height(null_at, order_by, hist_df)
    bin_width = _get_bin_width(num_hists)

    # Plotting functions
    _plot_all_histograms(bin_left,
                         bin_height,
                         null_bin_left,
                         null_bin_height,
                         bin_width
                        )
    _plot_xticks(null_at, bin_left, hist_df)

    if log:
        _plot_new_yticks(bin_height)


def plot_numeric_hists(df_list, labels=[], nbins=25, log=False, normed=False,
                       null_at='left',
                       color_palette=sns.color_palette('deep')):
    """Plots numerical histograms together.
    
    Inputs:
    df_list - A pandas DataFrame or a list of DataFrames which have two
              columns (bin_nbr and freq). The bin_nbr is the value of
              the histogram bin and the frequency is how many values
              fall in that bin.
    labels - A string (for one histogram) or list of strings which sets
             the labels for the histograms
    nbins - The desired number of bins (Default: 25)
    log - Boolean of whether to display y axis on log scale
          (Default: False)
    normed - Boolean of whether to normalize histograms so that the
             heights of each bin sum up to 1. This is useful for
             plotting columns with difference sizes (Default: False)
    null_at - Which side to set a null value column. Options are 'left'
              or 'right'. Leave it empty to not include (Default: left)
    color_palette - Seaborn colour palette, i.e., a list of tuples
                    representing the colours. (Default: sns deep color
                    palette)
    """
    
    def _check_for_nulls(df_list):
        """Returns a list of whether each list has a null column."""
        return [df['bin_nbr'].isnull().any() for df in df_list]

    def _get_null_weights(has_null, df_list):
        """ If there are nulls, determine the weights.  Otherwise, set 
        weights to 0.
        
        Returns the list of null weights.
        """

        return [float(df[df['bin_nbr'].isnull()].weights)
                if is_null else 0 
                for is_null, df in zip(has_null, df_list)]

    def _get_data_type(bin_nbrs):
        """ Returns the data type in the histogram, i.e., whether it is
        numeric or a timetamp. This is important because it determines
        how we deal with the bins.
        """

        if 'float' in str(type(bin_nbrs[0][0])) or 'int' in str(type(bin_nbrs[0][0])):
            return 'numeric'
        elif str(type(bin_nbrs[0][0])) == "<class 'pandas.tslib.Timestamp'>":
            return 'timestamp'
        else:
            raise Exception('Bin data type not valid: {}'.format(type(bin_nbrs[0][0])))

    def _plot_hist(data_type, bin_nbrs, weights, labels, bins, log):
        """Plots the histogram for non-null values with corresponding
        labels if provided. This function will take also reduce the
        number of bins in the histogram. This is useful if we want to
        apply get_histogram_values for a large number of bins, then 
        experiment with plotting different bin amounts using the
        histogram values.
        """

        # If the bin type is numeric
        if data_type == 'numeric':
            if len(labels) > 0:
                _, bins, _ = plt.hist(x=bin_nbrs, weights=weights,
                                      label=labels, bins=nbins, log=log)
            else:
                _, bins, _ = plt.hist(x=bin_nbrs, weights=weights, bins=nbins,
                                      log=log)
            return bins

        # If the bin type is datetime or a timestamp
        elif data_type == 'timestamp':
            # Since pandas dataframes will convert timestamps and date
            # types to pandas.tslib.Timestamp types, we will need
            # to convert them to datetime since these can be plotted.
            datetime_list = [dt.to_pydatetime() for dt in bin_nbrs[0]]
            _, bins, _ = plt.hist(x=datetime_list, weights=weights[0],
                                  bins=nbins, log=log)
            return bins

    def _get_null_bin_width(data_type, bin_info, num_hists, null_weights):
        """Returns the width of each null bin."""
        bin_width = bin_info[1] - bin_info[0]
        if num_hists == 1:
            return bin_width
        else:
            return 0.8 * bin_width/len(null_weights)

    def _get_null_bin_left(data_type, loc, num_hists, bin_info, null_weights):
        """Gets the left index/indices or the null column(s)."""
        bin_width = bin_info[1] - bin_info[0]
        if loc == 'left':
            if num_hists == 1:
                return [bin_info[0] - bin_width]
            else:
                return [bin_info[0] - bin_width + bin_width*0.1 + i*_get_null_bin_width(data_type, bin_info, num_hists, null_weights) for i in range(num_hists)]
        elif loc == 'right':
            if num_hists == 1:
                return [bin_info[-1]]
            else:
                return [bin_width*0.1 + i*_get_null_bin_width(data_type, bin_info, num_hists, null_weights) + bin_info[-1] for i in range(num_hists)]
        elif loc == 'order':
            raise Exception('null_at = order is not supported for numeric histograms.')

    def _plot_null_xticks(loc, bins, xticks):
        """Given current xticks, plot appropriate NULL tick."""
        bin_width = bins[1] - bins[0]
        if loc == 'left':
            plt.xticks([bins[0] - bin_width*0.5] + xticks[1:].tolist(), ['NULL'] + [int(i) for i in xticks[1:]])
        elif loc == 'right':
            plt.xticks(xticks[:-1].tolist() + [bins[-1] + bin_width*0.5], [int(i) for i in xticks[:-1]] + ['NULL'])

    def _get_xlim(loc, has_null, bins, null_bin_left, null_bin_height):
        """Gets the x-limits for plotting."""
        if loc == '' or not np.any(has_null):
            # If we do not want to plot nulls, or if there are no nulls
            # in the data, then set the limits as the regular histogram
            # limits
            xlim_left = bins[0]
            xlim_right = bins[-1]
        else:
            xlim_left = min(bins.tolist() + null_bin_left)
            if loc == 'left':
                xlim_right = max(bins.tolist() + null_bin_left)
            elif loc == 'right':
                xlim_right = max(bins.tolist() + null_bin_left) + null_bin_height

        return xlim_left, xlim_right


    df_list, labels = _listify(df_list, labels)
    # Joins in all the df_list DataFrames
    # Number of histograms we want to overlay
    num_hists = len(df_list)

    # If any of the columns are null
    has_null = _check_for_nulls(df_list)
    _add_weights_column(df_list, normed)

    # Set color_palette
    sns.set_palette(color_palette)
    null_weights = _get_null_weights(has_null, df_list)
    
    df_list = [df.dropna() for df in df_list]
    weights = [df.weights for df in df_list]
    bin_nbrs = [df.bin_nbr for df in df_list]
    
    data_type = _get_data_type(bin_nbrs)

    # Plot histograms and retrieve bins
    bin_info = _plot_hist(data_type, bin_nbrs, weights, labels, nbins, log)

    null_bin_width = _get_null_bin_width(data_type, bin_info, num_hists, null_weights)
    null_bin_left = _get_null_bin_left(data_type, null_at, num_hists, bin_info, null_weights)
    xticks, _ = plt.xticks()

    # If we are plotting NULLS and there are some, plot them and change xticks
    if null_at != '' and np.any(has_null):
        for i in range(num_hists):
            plt.bar(null_bin_left[i], null_weights[i], null_bin_width,
                    color=color_palette[i], hatch='x')
        if data_type == 'numeric':
            _plot_null_xticks(null_at, bin_info, xticks)
        elif data_type == 'timestamp':
            pass 
    # Set the x axis limits
    plt.xlim(_get_xlim(null_at, has_null, bin_info, null_bin_left, null_bin_width))


def plot_date_hists(df_list, labels=[], nbins=25, log=False, normed=False,
                    null_at='left', color_palette=sns.color_palette('deep')):
    """Plots histograms by date.

    Inputs:
    df_list - A pandas DataFrame or a list of DataFrames which have two
              columns (bin_nbr and freq). The bin_nbr is the value of
              the histogram bin and freq is how many values fall in that
              bin.
    labels - A string (for one histogram) or list of strings which sets
             the labels for the histograms
    nbins - The desired number of bins (Default: 25)
    log - Boolean of whether to display y axis on log scale
          (Default: False)
    normed - Boolean of whether to normalize histograms so that the
             heights of each bin sum up to 1. This is useful for
             plotting columns with difference sizes (Default: False)
    null_at - Which side to set a null value column. Options are 'left'
              or 'right'. Leave it empty to not include (Default: left)
    color_palette - Seaborn colour palette, i.e., a list of tuples
                    representing the colours. (Default: sns deep color
                    palette)
    """

    df_list, labels = _listify(df_list, labels)
    # Joins in all the df_list DataFrames
    # Number of histograms we want to overlay
    num_hists = len(df_list)

    # If any of the columns are null
    has_null = _check_for_nulls(df_list)
    _add_weights_column(df_list, normed)

    # Set color_palette
    sns.set_palette(color_palette)
    null_weights = _get_null_weights(has_null, df_list)

    print has_null
    print null_weights


def plot_scatterplot(scatter_df, s=20, c=sns.color_palette('deep')[0],
                     plot_type='scatter', by_size=True, by_opacity=True,
                     marker='o'):
    """Plots a scatter plot based on the computed scatter plot bins.

    Inputs:
    scatter_df - a pandas DataFrame which has three columns (scat_bin_x,
                 scat_bin_y, and freq), where the scat_bin_x and
                 scat_bin_y are the bins along the x and y axes and freq
                 is how many values fall in that bin.
    s - The size of each point (Default: 20)
    c - The colour of the plot (Default: seaborn deep blue)
    plot_type - The plot type. Can be either 'scatter' or 'heatmap'.
                (Default: scatter)
    by_size - If True, then the size of each plotted point will be
              proportional to the frequency. Otherwise, it will be a
              constant size specified by s (Default: True)
    by_opacity - If True, then the opacity of each plotted point will be
                 proportional to the frequency. Darker implies more data
                 in that bin. (Default: True)
    marker - matplotlib marker to plot (Default: 'o')
    """

    if plot_type == 'scatter':
        if not by_size and not by_opacity:
            raise Exception('Scatterplot must be plotted by size and/or opacity.')

        if by_size:
            plot_size = 20*scatter_df.freq
        else:
            plot_size = 20

        if by_opacity:
            colour = np.zeros((scatter_df.shape[0], 4))
            colour[:, :3] = c
            # Add alpha component
            colour[:, 3] = scatter_df.freq/scatter_df.freq.max()
            lw = 0 
        else:
            colour = c
            lw = 0.5

        plt.scatter(scatter_df.scat_bin_x, scatter_df.scat_bin_y,
                    c=colour, s=plot_size, lw=lw, marker=marker)

    elif plot_type == 'heatmap':
        num_x = len(scatter_df.scat_bin_x.value_counts())
        num_y = len(scatter_df.scat_bin_y.value_counts())

        x = scatter_df['scat_bin_x'].values.reshape(num_x, num_y)
        y = scatter_df['scat_bin_y'].values.reshape(num_x, num_y)
        z = scatter_df['freq'].values.reshape(num_x, num_y) 

        plt.pcolor(x, y, z)
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())
