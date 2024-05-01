all_variables = [
    "Account",
    "Comp_Val",
    "Location",
    "Outlets",
    "Cust_Int",
    "Advertise",
    "Shirts_Price",
    "Shirts_Sales",
    "Shirts_Stock",
    "Raw_Stock",
    "Raw_Price",
    "Raw_Order",
    "Worker_50",
    "Worker_100",
    "Worker_Sat",
    "Salary",
    "Social_Cost",
    "Repair",
    "Mach_50",
    "Mach_100",
    "Prod_Idle",
    "Damage"
]

financeMap = {
    'income_Outlets'         : "Outlets",
    'income_Machines50'      : "Machines 50",
    'income_Machines100'     : "Machines 100",
    'income_Sales'           : "Sales",
    'income_Interest'        : "Interest",
    'income_savings'         : "Savings/Loan",
    'investments_Outlets'    : "Outlets",
    'investments_Machines50' : "Machines 50",
    'investments_Machines100': "Machines 100",
    'expenses_Material'      : "Material",
    'expenses_Salary'        : "Salary",
    'expenses_Social'        : "Social Costs",
    'expenses_Outlets'       : "Outlets",
    'expenses_Location'      : "Location",
    'expenses_Storage'       : "Storage",
    'expenses_Advertising'   : "Advertising",
    'expenses_MachineService': "Repair",
    'expenses_Interest'      : "Interest",
}

ts_replacement_map = {
    "Workers50":          "Worker_50",
    "Workers100":         "Worker_100",
    "WorkerSalary":       "Salary",
    "ShirtPrice":         "Shirts_Price",
    "SalesOutlets":       "Outlets",
    "MaterialOrder":      "Raw_Order",
    "Machines50":         "Mach_50",
    "Machines100":        "Mach_100",
    "MachineService":     "Repair",
    "WorkerBenefits":     "Social_Cost",
    "Advertising":        "Advertise",
    "BusinessLocation":   "Location",
    "BankAccount":        "Account",
    "ShirtSales":         "Shirts_Sales",
    "MaterialPrice":      "Raw_Price",
    "ShirtStock":         "Shirts_Stock",
    "WorkerSatisfaction": "Worker_Sat",
    "ProductionIdle":     "Prod_Idle",
    "CompanyValue":       "Comp_Val",
    "CustomerInterest":   "Cust_Int",
    "MaterialStock":      "Raw_Stock",
    "MachineCapacity":    "Damage",
}

node_replacement_map = {
    "Kontostand" : "Account",
    "Unternehmenswert" : "Comp_Val",
    "Geschäftsstandort" : "Location",
    "Verkaufsstellen" : "Outlets",
    "Kundeninteresse" : "Cust_Int",
    "Werbeausgaben" : "Advertise",
    "Hemdenpreis" : "Shirts_Price",
    "Hemden verkauft" : "Shirts_Sales",
    "Hemden auf Lager" : "Shirts_Stock",
    "Rohmaterial auf Lager" : "Raw_Stock",
    "Preis Rohmaterial" : "Raw_Price",
    "Rohmaterial Bestellung " : "Raw_Order",
    "Arbeiter 50" : "Worker_50",
    "Arbeiter 100" : "Worker_100",
    "Arbeitszufriedenheit" : "Worker_Sat",
    "Lohn" : "Salary",
    "Sozialkosten pro Arb." : "Social_Cost",
    "Reparatur & Service" : "Repair",
    "50er Maschinen" : "Mach_50",
    "100er Maschinen" : "Mach_100",
    "Produktionsausfall" : "Prod_Idle",
    "Maschinenschäden" : "Damage",
}

controllable_variables = [
    "Worker_50",
    "Worker_100",
    "Salary",
    "Shirts_Price",
    "Outlets",
    "Raw_Order",
    "Mach_50",
    "Mach_100",
    "Repair",
    "Social_Cost",
    "Advertise",
    "Location"
]

derived_variables = [
    "Account",
    "Shirts_Sales",
    "Raw_Price",
    "Shirts_Stock",
    "Worker_Sat",
    "Prod_Idle",
    "Comp_Val",
    "Cust_Int",
    "Raw_Stock",
    "Damage"
]

default_values = {
    "Worker_50":        8,
    "Worker_100":       0,
    "Salary":     1080,
    "Shirts_Price":       52,
    "Outlets":     1,
    "Raw_Order":    0,
    "Mach_50":       10,
    "Mach_100":      0,
    "Repair":   1200,
    "Social_Cost":   50,
    "Advertise":      2800,
    "Location": 1,
    "Account":        165775,
    "Shirts_Sales":         407,
    "Raw_Price":      4,
    "Shirts_Stock":         81,
    "Worker_Sat": 0.98,
    "Prod_Idle":     0.0,
    "Comp_Val":       250691,
    "Cust_Int":   767,
    "Raw_Stock":      16,
    "Damage":    47
}

edges = [
    # Damage
    ("Repair", "-", "Damage"), ("Mach_50", "+", "Damage"), ("Mach_100", "+", "Damage"),
    # Worker_ Satisfaction
    ("Salary", "+", "Worker_Sat"), ("Social_Cost", "+", "Worker_Sat"),
    # ProductionIdle (since this is ratio of lost potential, it gets increased with more production capacity)
    ("Raw_Stock", "-", "Prod_Idle"), ("Worker_Sat", "+", "Prod_Idle"), ("Mach_50", "+", "Prod_Idle"), ("Mach_100", "+", "Prod_Idle"), ("Worker_50", "+", "Prod_Idle"), ("Worker_100", "+", "Prod_Idle"),
    # Material in stock (Damage to machines prevents the stock from being depleted - but it cannot be expressed by a positive relationship either...)
    ("Worker_Sat", "-", "Raw_Stock"), ("Mach_50", "-", "Raw_Stock"), ("Mach_100", "-", "Raw_Stock"), ("Worker_50", "-", "Raw_Stock"), ("Worker_100", "-", "Raw_Stock"), ("Raw_Order", "+", "Raw_Stock"),
    # Shirt sales
    ("Shirts_Stock", "+", "Shirts_Sales"), ("Cust_Int", "+", "Shirts_Sales"), ("Shirts_Price", "-", "Shirts_Sales"),
    # Shirt stock
    ("Shirts_Sales", "-", "Shirts_Stock"), ("Worker_Sat", "+", "Shirts_Stock"), ("Mach_50", "+", "Shirts_Stock"), ("Mach_100", "+", "Shirts_Stock"), ("Worker_50", "+", "Shirts_Stock"), ("Worker_100", "+", "Shirts_Stock"), ("Raw_Stock", "+", "Shirts_Stock"), ("Damage", "-", "Shirts_Stock"),
    # Customer Interest
    ("Advertise", "+", "Cust_Int"), ("Location", "+", "Cust_Int"), ("Outlets", "+", "Cust_Int"),
    # Account (Interest would be an edge to itself, which is omitted here)
    # Regular expenses
    ("Raw_Price", "-", "Account"), ("Raw_Order", "-", "Account"), 
    ("Salary", "-", "Account"), ("Social_Cost", "-", "Account"), ("Worker_50", "-", "Account"), ("Worker_100", "-", "Account"),
    ("Outlets", "-", "Account"), ("Location", "-", "Account"),
    ("Shirts_Stock", "-", "Account"),
    ("Repair", "-", "Account"), ("Advertise", "-", "Account"),
    # Investments (outlets are investments, but are already covered)
    ("Mach_50", "-", "Account"), ("Mach_100", "-", "Account"),
    # Revenue
    ("Shirts_Sales", "+", "Account"), ("Shirts_Price", "+", "Account"),
    # Company value
    ("Mach_50", "+", "Comp_Val"), ("Mach_100", "+", "Comp_Val"), ("Damage", "-", "Comp_Val"),
    ("Outlets", "+", "Comp_Val"), 
    ("Raw_Stock", "+", "Comp_Val"), ("Shirts_Stock", "+", "Comp_Val"),
    ("Account", "+", "Comp_Val")
]
