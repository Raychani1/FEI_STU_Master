package passwordsecurity2;

import passwordsecurity2.Database.MyResult;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.StringTokenizer;
import java.util.concurrent.TimeUnit;

public class Login {
    protected static MyResult prihlasovanie(String meno, String heslo) throws IOException, Exception {
        /*
         *   Delay je vhodne vytvorit este pred kontolou prihlasovacieho mena.
         */
        TimeUnit.MILLISECONDS.sleep(100);

        MyResult account = Database.find(meno);
        if (!account.getFirst()) {
            return new MyResult(false, "Nespravne meno.");
        } else {
            StringTokenizer st = new StringTokenizer(account.getSecond(), ":");
            st.nextToken();
            String password = st.nextToken();
            String salt = st.nextToken();

            /*
             *   Pred porovanim hesiel je nutne k heslu zadanemu od uzivatela pridat prislusny salt z databazy a nasledne tento retazec zahashovat.
             */
            byte[] decodedSalt = Base64.getDecoder().decode(salt);
            String hashedHeslo = Security.hash(heslo, decodedSalt);

            boolean rightPassword = password.equals(Base64.getEncoder().encodeToString(hashedHeslo.getBytes(StandardCharsets.UTF_8)));

            if (!rightPassword)
                return new MyResult(false, "Nespravne heslo.");
        }
        return new MyResult(true, "Uspesne prihlasenie.");
    }
}