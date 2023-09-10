hys_df <- read.csv("D:\\Google 드라이브\\데이터 분석\\미팅자료\\21.02.03 히스테리시스\\attachments\\pn500.csv")

eval_func = function(vec, gamma, beta, alpha, n){
    
    
    dx = diff(vec)
    # gamma = 0.1;
    # beta = 0.5;
    # alpha = 0.001;
    # n = 1;
    
    
    eval_inc <- function(z, dx, gamma, beta, alpha, n) {
        psi = gamma + beta * sign(dx * z);
        dz = dx * (1 - abs(z)^n * psi);
        df = (1 - alpha) * dx + alpha * dz;
        de = z * dx
        
        ret <- list(
            dz = dz,
            df = df,
            de = de
        )
    }
    #plot(df)
    
    z = matrix(0,length(dx),1)
    x = matrix(0,length(dx),1)
    f = matrix(0,length(dx),1)
    e = matrix(0,length(dx),1)
    #df = matrix(0,length(dx),1)
    i = 1
    while( i < length(dx) ){
        res = eval_inc(z[i,1], dx, gamma, beta, alpha, n)
        
        x[i+1,1] = x[i,1] + dx[i];
        z[i+1,1] = z[i,1] + res$dz[i];
        f[i+1,1] = f[i,1] + res$dz[i];
        e[i+1,1] = e[i,1] + res$de[i];
        #df[i] = res$df
        i = i + 1;
    }
    
    ret <- list(
        x = x,
        z = z,
        f = f,
        e= e
    )
    return(ret)
    
}
#############################################

a <- eval_func(hys_df[,2], 0.1, 0.5, 0.01, 1)

