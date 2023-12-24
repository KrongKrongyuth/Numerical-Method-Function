import numpy as np
import pandas as pd
import sympy as smp
x = smp.symbols("x", real = True) # Define the variable before run the class --> in this case i defind only x variable

class Numerical_Function :
    def __init__(self, function) : # You need to input the function follow by smypy syntax.
        self.fx = function
    
    def change_function(self, function) :
        self.__init__(function)
    
    def get_function(self, function = None) :
        if function == None :
            return self.fx 
        elif function != None :
            return function
        
    def get_diff_function(self, order = 1, function = None) :
        if function == None :
            df_dx = smp.diff(self.fx, x, order)
            return df_dx
        elif function != None :
            df_dx = smp.diff(function, x, order)
            return df_dx
        
    def evaluate_function(self, x_value, function = None) :
        if function == None :
            result = (self.fx).evalf(subs={x : x_value})
            return result
        elif function != None :
            result = function.evalf(subs={x : x_value})
            return result

    def evaluate_diff_function(self, x_value, order = 1, function = None) :
        if function == None :
            df_dx = smp.diff(self.fx, x, order)
            result = float(df_dx.evalf(subs={x : x_value}))
            return result
        elif function != None :
            df_dx = smp.diff(function, x, order)
            result = float(df_dx.evalf(subs={x : x_value}))
            return result
    
    def summary(self, function, df, method) :
        if method == "newton" or method == "Newton" :
            iteration = len(df) - 1
            x_n = df["x(n)"][iteration]
            f_x = self.get_function()
            df_x = self.get_diff_function()
            g_x = function

            n = 1
            order, result = 0, 0
            while True :
                result = self.evaluate_diff_function(x_n, order = n, function = g_x)
                if result > 10**-2 :
                    order = n
                    break
                n += 1
            x_n = round(x_n,6)
            result = round(result,6)
            final_info = print("Your iteration : {} \nYour result x(n) : {} \nYour order is {} with value {}".format(iteration, x_n, order, result))
            return final_info
        elif method == "bisection" or method == "Bisection" : # <-- This function must be done
            pass
        elif method == "fixedpoint" or method == "Fixedpoint" :
            iteration = len(df) - 1
            x_n = df["x(n)"][iteration]
            f_x = self.get_function()
            df_x = self.get_diff_function()
            g_x = function

            n = 1
            order, result = 0, 0
            while True :
                result = self.evaluate_diff_function(x_n, order = n, function = g_x)
                if result > 10**-2 :
                    order = n
                    break
                n += 1
            x_n = round(x_n,6)
            result = round(result,6)
            final_info = print("Your iteration : {} \nYour result x(n) : {} \nYour order is {} with value {}".format(iteration, x_n, order, result))
            return final_info
        elif method == "secant" or method == "Secant" : # <-- This function must be done
            pass
        elif method == "mullers" or method == "Mullers" : # <-- This function must be done
            pass
    
    def bisection(self, interval, TOL = None, round = None) :
        # Create DataFrame
        bisection_df = pd.DataFrame({"x(n)" : [None],
                                     "a(n)" : interval[0],
                                     "b(n)" : interval[1],
                                     "|x(n-1)-x(n)|" : np.abs(interval[0] - interval[1]),
                                     "f(x(n))" : [None],
                                     "f(a(n))" : [None],
                                     "f(b(n))" : [None]})
        
        # Start loop
        n = 1
        while True :
            x_n = (bisection_df["a(n)"][n-1] + bisection_df["b(n)"][n-1])/2
            f_an = self.evaluate_function(bisection_df["a(n)"][n-1])
            f_bn = self.evaluate_function(bisection_df["b(n)"][n-1])
            f_xn = self.evaluate_function(x_n)                                             
            error = (bisection_df["|x(n-1)-x(n)|"][n-1])/2
            if f_xn == 0 :
                return print("p is equal {}".format(x_n))
            elif f_an*f_xn < 0 :
                a_n = bisection_df["a(n)"][n-1]
                b_n = x_n
            elif f_xn*f_bn < 0 :
                a_n = x_n
                b_n = bisection_df["b(n)"][n-1]
            new_row = {"x(n)" : x_n,
                       "a(n)" : a_n,
                       "b(n)" : b_n,
                       "|x(n-1)-x(n)|" : error,
                       "f(x(n))" : f_xn,
                       "f(a(n))" : f_an,
                       "f(b(n))" : f_bn}
            bisection_df = bisection_df.append(new_row, ignore_index = True)
            if TOL == None :
                if round == n : 
                    return bisection_df
            else :
                if bisection_df["|x(n-1)-x(n)|"][n] < TOL :
                    return bisection_df
            n += 1

    def fixedpoint(self, x_0, TOL = None, round = None) : # When you need to use Fixed Point method you must change the function first
        # Create DataFrame
        fixedpoint_df = pd.DataFrame({"x(n)" : [x_0],
                                      "|x(n)-x(n-1)|" : [1],
                                      "error(n)/error(n-1)" : [None]})
        # Start loop
        n = 1
        while True :
            x_n_old = fixedpoint_df["x(n)"][n-1]
            x_n = self.evaluate_function(x_n_old)
            error = np.abs(x_n - x_n_old)
            error_ratio = error/fixedpoint_df["|x(n)-x(n-1)|"][n-1]
            new_row = {"x(n)" : x_n,
                       "|x(n)-x(n-1)|" : error,
                       "error(n)/error(n-1)" : error_ratio}
            fixedpoint_df = fixedpoint_df.append(new_row, ignore_index = True)
            if TOL == None :
                if round == n : 
                    return fixedpoint_df
            else :
                if fixedpoint_df["|x(n)-x(n-1)|"][n] < TOL :
                    return fixedpoint_df
            n += 1
    
    def newton(self, x_0, TOL = None, round = None) :
        # Create DataFrame
        newton_df = pd.DataFrame({"x(n)" : [x_0],
                                  "f(x(n))" : [None],
                                  "df/dx(x(n))" : [None],
                                  "|x(n)-x(n-1)|" : [None]})
        
        # Start loop
        n = 1
        while True :
            x_n_old = newton_df["x(n)"][n-1]
            f_xn = self.evaluate_function(x_n_old)
            df_xn = self.evaluate_diff_function(x_n_old)
            x_n_new = x_n_old - (f_xn/df_xn)
            error = np.abs(x_n_new - x_n_old)
            new_row = {"x(n)" : x_n_new,
                       "f(x(n))" : f_xn,
                       "df/dx(x(n))" : df_xn,
                       "|x(n)-x(n-1)|" : error}
            newton_df = newton_df.append(new_row, ignore_index = True)
            if TOL == None :
                if round == n : 
                    return newton_df
            else :
                if newton_df["|x(n)-x(n-1)|"][n] < TOL :
                    return newton_df
            n += 1
    
    def secant(self, x_value, TOL = None, round = None) : # x_value must be list of [x0, x1]
        # Calculate basic term
        x_n = x_value[1]
        x_n_old = x_value[0]
        f_xn = self.evaluate_function(x_n)
        f_xn_old = self.evaluate_function(x_n_old)
        error = np.abs(x_n - x_n_old)

        # Create DataFrame
        secant_df = pd.DataFrame({"x(n)" : [x_n],
                                  "x(n-1)" : [x_n_old],
                                  "f(x(n)" : [f_xn],
                                  "f(x(n-1))" : [f_xn_old],
                                  "|x(n)-x(n-1)|" : [error]})
        
        # Start loop
        n = 1
        while True :
            x_n_old = secant_df["x(n-1)"][n-1] 
            x_n = secant_df["x(n)"][n-1]
            f_xn_old = self.evaluate_function(x_n_old)
            f_xn = self.evaluate_function(x_n)
            x_n_new = x_n - (((x_n-x_n_old)/(f_xn - f_xn_old))*(f_xn))
            error = np.abs(x_n_new - x_n)
            new_row = {"x(n)" : x_n_new,
                       "x(n-1)" : x_n,
                       "f(x(n)" : f_xn,
                       "f(x(n-1))" : f_xn_old,
                       "|x(n)-x(n-1)|" : error}
            secant_df = secant_df.append(new_row, ignore_index = True)
            if TOL == None :
                if round == n : 
                    return secant_df
            else :
                if secant_df["|x(n)-x(n-1)|"][n] < TOL :
                    return secant_df
            n += 1
    
    def mullers(self, x_value, TOL = None, round = None) : # x_value must be list of [x0, x1, x2]
        mullers_df = pd.DataFrame({"x(n)" : x_value[0],
                                   "x(n-1)" : x_value[1],
                                   "x(n-2)" : x_value[2],
                                   "|x(n)-x(n-1)|" : [None]})
        n = 0
        while True :
            # Calculate for the basic term
            if n == 0 :
                x0, x1, x2 = x_value[0], x_value[1], x_value[2]
            else :
                x0, x1, x2 = mullers_df["x(n-2)"][n-1], mullers_df["x(n-1)"][n-1], mullers_df["x(n)"][n-1],

            f_x0 = self.evaluate_function(x0)
            f_x1 = self.evaluate_function(x1)
            f_x2 = self.evaluate_function(x2)
            c = f_x2

            # Calculate the component of matrix A
            a_12 = x0 - x2
            a_11 = a_12**2
            a_22 = x1 - x2
            a_21 = a_22**2

            # Calculate the component of matrix B
            b_11 = f_x0 - c
            b_21 = f_x1 - c

            # Create the matrix
            matrix_A = smp.Matrix([[a_11, a_12],[a_21, a_22]])
            try : # I found the case that matrix cannot find the invrese so when that case is appear I will return mullers_df
                matrix_A_inv = matrix_A.inv() # Find inverse matrix of A
            except :
                return mullers_df
            matrix_B = smp.Matrix([[b_11],[b_21]])

            # Slove for a,b
            matrix_X = matrix_A_inv * matrix_B # if we use sympy for create we can multiplie matrix by using *
            a = matrix_X[0]
            b = matrix_X[1]

            # Slove for x3
            if b < 0 :
                x3 = x2 - ((2*c)/(b - smp.sqrt(b**2 - (4*a*c))))
            else :
                x3 = x2 - ((2*c)/(b + smp.sqrt(b**2 - (4*a*c))))
                
            # Calculate for error
            error = np.abs(x3-x2)

            # Create the row for append into DataFrame
            new_row = {"x(n)" : x3,
                       "x(n-1)" : x2,
                       "x(n-2)" : x1,
                       "|x(n)-x(n-1)|" : error}
            if n == 0 :
                mullers_df.iloc[n] = new_row
            else :
                mullers_df = mullers_df.append(new_row, ignore_index = True)
            
            # return condition
            if TOL == None :
                if round == n : 
                    return mullers_df
            else :
                if mullers_df["|x(n)-x(n-1)|"][n] < TOL :
                    return mullers_df
            n += 1