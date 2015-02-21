downloadFile <- function (directory = NULL, dataset = NULL, url = NULL) {
 
    goTo <- url
    data <- file.path(directory,dataset)
    
    if (!file.exists(directory)) dir.create(directory)       
    
    if (!file.exists(data)) {
        download.file(goTo, destfile = data)
    }
    
}

getNaStrings <- function () {
    
    na.strings <- c("?","", "NA", "na", "N/A", "n/a", "NULL", "null", "NONE", "none","9999","#DIV/0!",
                    "#N/a")
    
    na.strings
    
}

sessInfo <- function (clear = TRUE, loc = c(), log = TRUE) {    
    
    if (length(loc) > 0) {
        
        Sys.setlocale(loc[1], loc[2])
        
    }
    
    if (clear == FALSE) {
        
        rm(list = ls(.GlobalEnv), envir = .GlobalEnv)
        
    }
    
    if (log == TRUE) {
        
        session <- sessionInfo()
        
        return(session)
        
    } else {
        
        return(FALSE) 
        
    }
    
}

trim <- function (file = NULL) {
    
    gsub("(^[[:space:]]+|[[:space:]]+$)", "", file)
    
}