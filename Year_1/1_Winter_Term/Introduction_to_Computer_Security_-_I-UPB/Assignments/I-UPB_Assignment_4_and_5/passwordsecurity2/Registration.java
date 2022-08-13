package passwordsecurity2;

import passwordsecurity2.Database.MyResult;

import java.nio.charset.StandardCharsets;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;


public class Registration {
    protected static MyResult registracia(String meno, String heslo) throws NoSuchAlgorithmException, Exception {
        if (Database.exist(meno)) {
            System.out.println("Meno je uz zabrate.");
            return new MyResult(false, "Meno je uz zabrate.");
        } else {
            MyResult validPassword = Security.isValid(heslo);

            if(!validPassword.getFirst())
                return validPassword;

            String name = meno;
            byte[] salt = Security.getSalt();
            String password = Security.hash(heslo, salt);


            Database.add(name + ":" + Base64.getEncoder().encodeToString(password.getBytes(StandardCharsets.UTF_8)) + ":" + Base64.getEncoder().encodeToString(salt));

        }
        return new MyResult(true, "");
    }

}
